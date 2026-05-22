"""Clinical judge runner — runs the full judge panel for one QA/rubric triple.

Three-stage scoring strategy
-----------------------------
Stage 1 — Batch pass
  All judges except SKIP_BATCH ones send a single prompt covering all rubric
  items.  Concurrent via ThreadPoolExecutor.

Stage 2 — Per-item retry
  Any item that still has an NA score after stage 1 gets its own focused
  prompt via adapter.build_item_messages().  SKIP_BATCH judges enter here
  directly.  Concurrent via ThreadPoolExecutor.

Stage 3 — Hard-split pass  (NEW in v4)
  Any item that STILL has an NA score after stage 2 gets one absolute-minimum
  prompt via adapter.build_split_item_messages() — contains only the scale,
  criterion name+description, Q (truncated), A (truncated), and a primed ID
  prefix.  This guarantees every model produces a numeric score for every
  rubric item, regardless of prompt complexity issues.

Routing
-------
  Completion models: POST /v1/completions
  MedGemma per-item / split: POST /v1/chat/completions (adapter.ITEM_ENDPOINT)

Pre-flight
----------
  Before running any row, verify >= 2 judges are reachable.
  If < 2 judges respond, skip row and log as 'skipped'.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.consensus_core.models import (
    Answer, JudgeScore, Question, Rubric, RubricItem, new_id,
)
from core.consensus_core.events import EventLog, append_judgment_recorded, append_agreement_classified
from core.consensus_core.repository import InMemoryStore
from core.rubric_engine import DynamicRubricParser
from core.agreement import classify_panel_agreement, summarize_agreement
from core.metrics import get_metrics_collector
from core.model_adapters import get_adapter, ENDPOINT_CHAT, ENDPOINT_COMPLETION

logger = logging.getLogger('clinical_judge_runner')


@dataclass
class JudgeResult:
    judge_id: str
    aggregate_score: float
    scores: List[Dict]
    raw_response: str
    latency_ms: float


@dataclass
class PanelResult:
    question_id: str
    question_text: str
    question_category: str
    rubric_id: str
    rubric_name: str
    rubric_source_paper: str
    judge_results: List[JudgeResult]
    agreement_summary: Dict
    agreement_class: str
    outlier_judge: Optional[str]
    mean_pairwise_agreement: float
    events_jsonl: str
    skipped: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def _is_na(score) -> bool:
    """Return True if a score value is effectively missing/NA."""
    return str(score).upper().strip() in ('', 'NA', 'N/A', 'NONE')


class ADRDJudgeRunner:
    AGREEMENT_THRESHOLD  = 80.0
    MIN_JUDGES_REQUIRED  = 2

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config: Dict[str, Any] = json.load(f)
        self.judges: List[Dict] = self.config['judges']
        self.metrics = get_metrics_collector()
        self.store   = InMemoryStore()
        logger.info(f'Clinical judge runner: {len(self.judges)} judges')
        for j in self.judges:
            logger.info(f'  {j["id"]} | {j["model"]} | {j["url"]}')

    # ------------------------------------------------------------------
    # Internal: call one judge
    # ------------------------------------------------------------------

    def _call_judge(
        self,
        judge: Dict,
        messages: List[Dict],
        extra: Dict,
        endpoint: str,
    ) -> Tuple[str, float]:
        import httpx
        t0       = time.time()
        base_url = judge['url']
        temp     = float(self.config.get('temperature', 0.0))

        if endpoint == ENDPOINT_COMPLETION:
            prompt_text = messages[0]['prompt']
            payload = {
                'model':       judge['model'],
                'prompt':      prompt_text,
                'temperature': temp,
                **extra,
            }
            url = f'{base_url}/v1/completions'
            resp = httpx.post(url, json=payload,
                              timeout=float(self.config.get('timeout_seconds', 120)))
            resp.raise_for_status()
            latency_ms = (time.time() - t0) * 1000
            return resp.json()['choices'][0]['text'], latency_ms

        else:  # ENDPOINT_CHAT
            payload = {
                'model':       judge['model'],
                'messages':    messages,
                'temperature': temp,
                **extra,
            }
            url = f'{base_url}/v1/chat/completions'
            resp = httpx.post(url, json=payload,
                              timeout=float(self.config.get('timeout_seconds', 120)))
            resp.raise_for_status()
            latency_ms = (time.time() - t0) * 1000
            return resp.json()['choices'][0]['message']['content'], latency_ms

    def _call_judge_safe(
        self,
        judge: Dict,
        messages: List[Dict],
        extra: Dict,
        endpoint: str,
    ) -> Tuple[str, str, float]:
        jid = judge['id']
        try:
            raw, lat = self._call_judge(judge, messages, extra, endpoint)
            return jid, raw, lat
        except Exception as e:
            logger.error(f'Judge {jid} failed: {e}')
            return jid, '', 0.0

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def _preflight_check(self) -> List[str]:
        import httpx
        live = []
        for judge in self.judges:
            try:
                r = httpx.get(f"{judge['url']}/v1/models", timeout=5.0)
                if r.status_code == 200:
                    live.append(judge['id'])
                else:
                    logger.warning(f'Judge {judge["id"]} /v1/models returned {r.status_code}')
            except Exception as e:
                logger.warning(f'Judge {judge["id"]} unreachable: {e}')
        logger.info(f'Pre-flight: {len(live)}/{len(self.judges)} judges live: {live}')
        return live

    # ------------------------------------------------------------------
    # Stage 2 helper: one per-item retry call
    # ------------------------------------------------------------------

    def _do_one_retry(
        self,
        judge_item: Tuple[Dict, RubricItem],
        question_text: str,
        answer_text: str,
    ) -> Tuple[str, str, str, float]:
        """Returns (judge_id, item_id, raw_response, latency_ms)."""
        judge, item = judge_item
        adapter  = get_adapter(judge['id'])
        msgs     = adapter.build_item_messages(item, question_text, answer_text)
        extra    = adapter.extra_params_item()
        endpoint = getattr(adapter, 'ITEM_ENDPOINT', adapter.ENDPOINT)
        try:
            raw_i, lat_i = self._call_judge(judge, msgs, extra, endpoint)
        except Exception as e:
            logger.warning(f'Per-item call failed for {judge["id"]}/{item.id}: {e}')
            return judge['id'], item.id, '', 0.0
        return judge['id'], item.id, raw_i, lat_i

    # ------------------------------------------------------------------
    # Stage 3 helper: one hard-split call (minimum possible prompt)
    # ------------------------------------------------------------------

    def _do_one_split(
        self,
        judge_item: Tuple[Dict, RubricItem],
        question_text: str,
        answer_text: str,
    ) -> Tuple[str, str, str, float]:
        """Returns (judge_id, item_id, raw_response, latency_ms).

        Uses adapter.build_split_item_messages() which sends the absolute
        minimum prompt: scale + criterion + Q (truncated) + A (truncated)
        + primed ID prefix.  Designed for models that fail on multi-item or
        even regular per-item prompts due to context-window pressure.
        """
        judge, item = judge_item
        adapter  = get_adapter(judge['id'])
        msgs     = adapter.build_split_item_messages(item, question_text, answer_text)
        extra    = adapter.extra_params_split()
        # Split pass always uses the same endpoint as per-item
        endpoint = getattr(adapter, 'ITEM_ENDPOINT', adapter.ENDPOINT)
        try:
            raw_i, lat_i = self._call_judge(judge, msgs, extra, endpoint)
        except Exception as e:
            logger.warning(f'Hard-split call failed for {judge["id"]}/{item.id}: {e}')
            return judge['id'], item.id, '', 0.0
        return judge['id'], item.id, raw_i, lat_i

    # ------------------------------------------------------------------
    # Public: run full panel
    # ------------------------------------------------------------------

    def run(
        self,
        question: Question,
        answer: Answer,
        rubric: Rubric,
    ) -> PanelResult:

        live_judges = self._preflight_check()
        if len(live_judges) < self.MIN_JUDGES_REQUIRED:
            logger.error(
                f'Only {len(live_judges)} judges live for Q={question.id} — skipping'
            )
            return PanelResult(
                question_id=question.id, question_text=question.text,
                question_category=question.category or 'unknown',
                rubric_id=rubric.id, rubric_name=rubric.name,
                rubric_source_paper=rubric.source_paper,
                judge_results=[], agreement_summary={},
                agreement_class='skipped', outlier_judge=None,
                mean_pairwise_agreement=0.0, events_jsonl='',
                skipped=True,
            )

        log           = EventLog()
        judge_results: List[JudgeResult]          = []
        all_scores:    Dict[str, List[JudgeScore]] = {}

        # ------------------------------------------------------------------ #
        # Stage 1: Batch pass                                                 #
        # ------------------------------------------------------------------ #
        judge_prompts:     Dict[str, Tuple[List[Dict], Dict, str]] = {}
        skip_batch_judges: List[str] = []
        for judge in self.judges:
            adapter = get_adapter(judge['id'])
            if getattr(adapter, 'SKIP_BATCH', False):
                skip_batch_judges.append(judge['id'])
                logger.info(
                    f'Judge {judge["id"]}: SKIP_BATCH=True — going straight to per-item mode'
                )
            else:
                messages = adapter.build_messages(rubric.items, question.text, answer.text)
                extra    = adapter.extra_params()
                judge_prompts[judge['id']] = (messages, extra, adapter.ENDPOINT)

        active_judges = [j for j in self.judges if j['id'] in live_judges]
        batch_judges  = [j for j in active_judges if j['id'] not in skip_batch_judges]
        raw_responses: Dict[str, Tuple[str, float]] = {}

        if batch_judges:
            with ThreadPoolExecutor(max_workers=len(batch_judges)) as pool:
                futures = {
                    pool.submit(
                        self._call_judge_safe,
                        judge,
                        judge_prompts[judge['id']][0],
                        judge_prompts[judge['id']][1],
                        judge_prompts[judge['id']][2],
                    ): judge
                    for judge in batch_judges
                }
                for future in as_completed(futures):
                    jid, raw, lat = future.result()
                    raw_responses[jid] = (raw, lat)
                    logger.info(f'Judge {jid} batch returned {len(raw)} chars in {lat:.0f}ms')

        # SKIP_BATCH judges: seed empty so all their items go to per-item
        for jid in skip_batch_judges:
            if jid in live_judges:
                raw_responses[jid] = ('', 0.0)

        # ------------------------------------------------------------------ #
        # Parse batch results; queue NA items for Stage 2                     #
        # ------------------------------------------------------------------ #
        parser = DynamicRubricParser(rubric)
        item_results_per_judge: Dict[str, Dict[str, JudgeScore]] = {}
        stage2_calls: List[Tuple[Dict, RubricItem]] = []

        for judge in active_judges:
            jid      = judge['id']
            raw, _   = raw_responses.get(jid, ('', 0.0))
            adapter  = get_adapter(jid)
            batch    = adapter.parse(raw, rubric.items, jid)
            item_results_per_judge[jid] = {s.rubric_item_id: s for s in batch}
            for item in rubric.items:
                if _is_na(item_results_per_judge[jid][item.id].score):
                    stage2_calls.append((judge, item))

        # ------------------------------------------------------------------ #
        # Stage 2: Per-item retry                                             #
        # ------------------------------------------------------------------ #
        if stage2_calls:
            n_skip = sum(1 for (j, _) in stage2_calls if j['id'] in skip_batch_judges)
            n_real = len(stage2_calls) - n_skip
            logger.info(
                f'Stage 2 per-item: {len(stage2_calls)} item(s) — '
                f'{n_skip} SKIP_BATCH, {n_real} genuine NA retries'
            )
            with ThreadPoolExecutor(max_workers=max(1, len(active_judges))) as pool:
                futures2 = [
                    pool.submit(self._do_one_retry, ji, question.text, answer.text)
                    for ji in stage2_calls
                ]
                for fut in as_completed(futures2):
                    jid, iid, raw_i, _ = fut.result()
                    if not raw_i.strip():
                        continue
                    adapter = get_adapter(jid)
                    item    = next(it for it in rubric.items if it.id == iid)
                    new_sc  = adapter.parse_item(raw_i, item, jid)
                    if not _is_na(new_sc.score):
                        item_results_per_judge[jid][iid] = new_sc

        # ------------------------------------------------------------------ #
        # Stage 3: Hard-split pass — one minimal call per still-NA item      #
        # ------------------------------------------------------------------ #
        stage3_calls: List[Tuple[Dict, RubricItem]] = []
        for judge in active_judges:
            jid = judge['id']
            for item in rubric.items:
                if _is_na(item_results_per_judge[jid][item.id].score):
                    stage3_calls.append((judge, item))

        if stage3_calls:
            logger.info(
                f'Stage 3 hard-split: {len(stage3_calls)} item(s) still NA — '
                f'sending minimum single-criterion prompts'
            )
            with ThreadPoolExecutor(max_workers=max(1, len(active_judges))) as pool:
                futures3 = [
                    pool.submit(self._do_one_split, ji, question.text, answer.text)
                    for ji in stage3_calls
                ]
                for fut in as_completed(futures3):
                    jid, iid, raw_i, _ = fut.result()
                    if not raw_i.strip():
                        logger.warning(
                            f'Hard-split returned empty for {jid}/{iid} — '
                            f'leaving as NA'
                        )
                        continue
                    adapter = get_adapter(jid)
                    item    = next(it for it in rubric.items if it.id == iid)
                    new_sc  = adapter.parse_item(raw_i, item, jid)
                    if not _is_na(new_sc.score):
                        item_results_per_judge[jid][iid] = new_sc
                        logger.info(
                            f'Hard-split recovered {jid}/{iid} = {new_sc.score}'
                        )
                    else:
                        logger.warning(
                            f'Hard-split could not recover {jid}/{iid} '
                            f'(raw: {raw_i[:60]!r}) — keeping NA'
                        )

        # ------------------------------------------------------------------ #
        # Build JudgeResult objects                                           #
        # ------------------------------------------------------------------ #
        judges_with_output: List[str] = []
        for judge in active_judges:
            jid      = judge['id']
            raw, lat = raw_responses.get(jid, ('', 0.0))
            adapter  = get_adapter(jid)

            judge_scores = [item_results_per_judge[jid][it.id] for it in rubric.items]
            agg          = parser.aggregate_score(judge_scores)
            rationales   = {s.rubric_item_id: (s.rationale or '') for s in judge_scores}

            scored_count = sum(1 for s in judge_scores if not _is_na(s.score))
            if scored_count > 0:
                judges_with_output.append(jid)

            print(f"\n{'='*60}")
            print(f'JUDGE: {jid} | RUBRIC: {rubric.name} | Q: {question.id}')
            print(f'Aggregate Score: {agg:.2f}')
            if raw:
                preview = raw[:300] + ('...' if len(raw) > 300 else '')
                print(f'Raw ({len(raw)} chars): {preview}')
            else:
                print('Raw: (per-item / split mode — no batch response)')
            for s in judge_scores:
                flag = ' \u26a0\ufe0f NA' if _is_na(s.score) else ''
                print(f'  [{s.rubric_item_id}] score={s.score}{flag} | '
                      f'{(s.rationale or "")[:120]}')
            print('='*60)

            all_scores[jid] = judge_scores
            append_judgment_recorded(log, question.id, rubric.id, jid,
                                     [s.id for s in judge_scores])

            jr = JudgeResult(
                judge_id=jid, aggregate_score=agg,
                scores=[{'item_id': s.rubric_item_id, 'score': s.score,
                         'rationale': s.rationale} for s in judge_scores],
                raw_response=raw, latency_ms=lat,
            )
            judge_results.append(jr)
            self.metrics.record_eval(
                question_id=question.id, rubric_id=rubric.id,
                rubric_name=rubric.name, judge_id=jid,
                aggregate_score=agg, rationales=rationales,
                latency_ms=lat, status='ok',
            )

        if len(judges_with_output) < len(active_judges):
            silent = [j['id'] for j in active_judges
                      if j['id'] not in judges_with_output]
            logger.warning(
                f'Q={question.id}: {len(judges_with_output)}/{len(active_judges)} '
                f'judges produced scored output after all 3 passes. '
                f'Silent judges: {silent}'
            )

        # ------------------------------------------------------------------ #
        # Pairwise agreement                                                  #
        # ------------------------------------------------------------------ #
        judge_ids = list(all_scores.keys())
        pairwise: Dict[Tuple[str, str], float] = {}
        for ja, jb in combinations(judge_ids, 2):
            sc = parser.calculate_pairwise_agreement(all_scores[ja], all_scores[jb])
            pairwise[(ja, jb)] = sc
            pairwise[(jb, ja)] = sc

        agreement_class, outlier = classify_panel_agreement(
            pairwise, judge_ids, self.AGREEMENT_THRESHOLD
        )
        summary  = summarize_agreement(pairwise, judge_ids, self.AGREEMENT_THRESHOLD)
        mean_pw  = summary['mean_pairwise_agreement']

        append_agreement_classified(log, question.id, rubric.id,
                                    agreement_class, outlier, mean_pw)
        self.store.put(question.id, log)
        self.metrics.record_agreement(
            question_id=question.id, rubric_id=rubric.id,
            judge_a=judge_ids[0] if judge_ids else '',
            judge_b=judge_ids[1] if len(judge_ids) > 1 else '',
            agreement_score=mean_pw, agreement_class=agreement_class,
        )

        if outlier:
            print(f'\n\u26a0\ufe0f  OUTLIER JUDGE: {outlier}')
            for jr in judge_results:
                if jr.judge_id == outlier:
                    print(f'   Score: {jr.aggregate_score:.2f}')
                    for s in jr.scores:
                        print(f'   [{s["item_id"]}] {(s["rationale"] or "")[:120]}')

        config_max = self.config.get('max_tokens')
        if config_max:
            for judge in active_judges:
                adapter_max = get_adapter(judge['id']).MAX_NEW_TOKENS
                if adapter_max != config_max:
                    logger.warning(
                        f'Judge {judge["id"]}: config max_tokens={config_max} '
                        f'ignored — adapter MAX_NEW_TOKENS={adapter_max} used'
                    )

        return PanelResult(
            question_id=question.id, question_text=question.text,
            question_category=question.category or 'unknown',
            rubric_id=rubric.id, rubric_name=rubric.name,
            rubric_source_paper=rubric.source_paper,
            judge_results=judge_results, agreement_summary=summary,
            agreement_class=agreement_class, outlier_judge=outlier,
            mean_pairwise_agreement=mean_pw, events_jsonl=log.to_jsonl(),
            skipped=False,
        )
