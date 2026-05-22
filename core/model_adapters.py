"""Per-judge prompt adapters + permissive parser for the clinical-QA judge panel.

Recommended judge panel (selected by direct testing in
results/judge_candidate_evaluation.md):

| Judge id            | Model                                | Endpoint                  | Notes                              |
|---------------------|--------------------------------------|---------------------------|------------------------------------|
| medgemma            | google/medgemma-4b-it                | /v1/completions (batch)   | chat for per-item via ITEM_ENDPOINT|
| biomistral          | BioMistral/BioMistral-7B             | /v1/completions           | per-item only (SKIP_BATCH=True)    |
| meditron            | epfl-llm/meditron-7b                 | /v1/completions           | base LLaMA-2 CPT, primed output    |
| medalpaca           | medalpaca/medalpaca-7b               | /v1/completions           | LLaMA-2 instruct [INST] format     |

Three-stage scoring strategy:
  1. Batch pass: adapter builds ONE prompt asking judge to score every rubric
     item in the canonical 'ID: SCORE | reason' pipe format.
     Adapters with SKIP_BATCH=True bypass this and go directly to per-item.
  2. Per-item retry / SKIP_BATCH mode: one focused call per rubric item.
     Wrapper reads getattr(adapter, 'ITEM_ENDPOINT', adapter.ENDPOINT) so
     adapters that switch format for single items (e.g. MedGemma -> chat)
     route to the correct endpoint automatically.
  3. Hard-split pass: if any NA scores remain after stage 2, one absolute
     minimum prompt per item is sent — only scale + criterion + Q + A + ID
     prime.  This guarantees every model produces a score for every item.

Parser strategies (permissive, in order):
  0. First-line primed digit  primed output starts with digit/NA (no ID prefix)
  1. pipe format              U1: 1 | reason
  2. prose word               U1: Meets  /  U1: Does not meet
  3. bare digit               U1: 1
  4. paren'd digit            U1: (1)
  5. dash separator           U1 - 1
  6. positional digits        bare space-separated sequence (e.g. '1 0 1 0 1')

Fixes applied (v4 — 2026-05-20):
  All adapters:
    Issue 9: build_split_item_messages() added to every adapter.
             This is a stripped-down single-item prompt used by
             wrapper._do_hard_split_pass() when both batch and
             per-item retries still leave NA scores.
             extra_params_split() companion returns max_tokens=16.

  MedGemma (v3 fixes retained):
    Issue 1: ITEM_ENDPOINT = ENDPOINT_CHAT
    Issue 2: STOP extended with '\\nFINAL', '\\n```', '\\n\\n'
    Issue 3: MAX_NEW_TOKENS 128->350

  BioMistral (v3 fixes retained):
    Issue 4: per-item prompt primes digit, max_tokens=8
    Issue 5: rubric_item.description[:80] added

  Meditron (v3 fixes retained):
    Issue 6: abstract example IDs, example at top, no Items abbrev line

  MedAlpaca (v3 fixes retained):
    Issue 7: abstract example IDs + neutral 'ok.' rationale
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from core.consensus_core.models import JudgeScore, RubricItem, new_id


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENDPOINT_CHAT       = 'chat'         # POST /v1/chat/completions
ENDPOINT_COMPLETION = 'completion'   # POST /v1/completions  (raw prompt)


# ---------------------------------------------------------------------------
# Permissive scoring parser
# ---------------------------------------------------------------------------

PROSE_TO_BINARY = {
    'meets': '1', 'meet': '1', 'yes': '1', 'present': '1', 'y': '1',
    'true': '1', 'agree': '1', 'pass': '1', 'passes': '1',
    'does not meet': '0', 'not meet': '0', 'not meets': '0',
    'doesnt meet': '0', 'doesn t meet': '0',
    'no': '0', 'absent': '0', 'n': '0', 'false': '0',
    'disagree': '0', 'fail': '0', 'fails': '0',
}

PROSE_TO_LIKERT = {
    'poor': '1', 'very poor': '1', 'unacceptable': '1', 'terrible': '1',
    'below average': '2', 'fair': '2', 'mediocre': '2', 'subpar': '2',
    'average': '3', 'adequate': '3', 'acceptable': '3', 'moderate': '3', 'ok': '3',
    'good': '4', 'above average': '4', 'solid': '4', 'strong': '4',
    'excellent': '5', 'outstanding': '5', 'perfect': '5', 'great': '5',
}

PROSE_TO_SCORE: Dict[str, str] = {**PROSE_TO_LIKERT, **PROSE_TO_BINARY}


def _id_pattern(ids: List[str]) -> str:
    escaped = [re.escape(i) for i in ids]
    return r'(?<![A-Za-z0-9])(' + '|'.join(escaped) + r')(?![A-Za-z0-9])'


def _is_likert_rubric(rubric_items: List[RubricItem]) -> bool:
    return not all((it.scale or '').upper() == 'BINARY' for it in rubric_items)


def parse_pipe_format(
    raw: str,
    rubric_items: List[RubricItem],
    judge_id: str,
) -> List[JudgeScore]:
    """Multi-strategy parser. Returns one JudgeScore per rubric item."""
    ids           = [it.id for it in rubric_items]
    id_re_str     = _id_pattern(ids)
    rubric_binary = all((it.scale or '').upper() == 'BINARY' for it in rubric_items)
    found: Dict[str, Tuple[str, str]] = {}

    # 0) First-line primed digit
    first_item = rubric_items[0]
    m0 = re.match(
        r'^\s*(NA|N/A|[0-5])\s*(?:\|(.*))?$',
        raw.strip().splitlines()[0] if raw.strip() else '',
        re.IGNORECASE,
    )
    if m0 and first_item.id.upper() not in found:
        score0  = m0.group(1).upper().replace('/', '')
        reason0 = (m0.group(2) or '').strip()[:200] or '(primed-first-line)'
        found[first_item.id.upper()] = (score0, reason0)

    # 1) Pipe format
    re_pipe = re.compile(
        id_re_str + r'\s*[:\-=]\s*(NA|N/A|[0-5])\s*\|(.+)',
        re.IGNORECASE,
    )
    for m in re_pipe.finditer(raw):
        iid   = m.group(1).upper()
        score = m.group(2).upper().replace('/', '').strip()
        if iid not in found:
            found[iid] = (score, m.group(3).strip()[:200])

    # 2) Prose level
    prose_pattern = '|'.join(
        sorted((re.escape(w) for w in PROSE_TO_SCORE), key=len, reverse=True)
    )
    re_prose = re.compile(
        id_re_str + r'\s*[:\-=]\s*(' + prose_pattern + r')\b',
        re.IGNORECASE,
    )
    for m in re_prose.finditer(raw):
        iid  = m.group(1).upper()
        word = m.group(2).lower().strip()
        mapped = (PROSE_TO_BINARY if rubric_binary else PROSE_TO_LIKERT).get(word) \
              or PROSE_TO_SCORE.get(word)
        if iid not in found and mapped is not None:
            found[iid] = (mapped, '(prose-mapped)')

    # 3) Bare digit
    re_bare = re.compile(
        id_re_str + r'\s*[:=]\s*(NA|N/A|[0-5])(?![0-9])',
        re.IGNORECASE,
    )
    for m in re_bare.finditer(raw):
        iid   = m.group(1).upper()
        score = m.group(2).upper().replace('/', '').strip()
        if iid not in found:
            found[iid] = (score, '(extracted)')

    # 4) Paren'd digit
    re_paren = re.compile(
        id_re_str + r'\s*[:=]?\s*\(\s*(NA|N/A|[0-5])\s*\)',
        re.IGNORECASE,
    )
    for m in re_paren.finditer(raw):
        iid   = m.group(1).upper()
        score = m.group(2).upper().replace('/', '').strip()
        if iid not in found:
            found[iid] = (score, '(paren)')

    # 5) Dash separator
    re_dash = re.compile(
        id_re_str + r'\s+[-\u2014\u2013]\s+(NA|N/A|[0-5])(?![0-9])',
        re.IGNORECASE,
    )
    for m in re_dash.finditer(raw):
        iid   = m.group(1).upper()
        score = m.group(2).upper().replace('/', '').strip()
        if iid not in found:
            found[iid] = (score, '(dash)')

    # 6) Positional fallback
    n_items = len(rubric_items)
    if len(found) < n_items:
        for line in raw.strip().splitlines():
            tokens = line.strip().split()
            if len(tokens) == n_items and all(
                re.fullmatch(r'[0-5]|NA|N/A', t, re.IGNORECASE) for t in tokens
            ):
                for it, tok in zip(rubric_items, tokens):
                    if it.id.upper() not in found:
                        found[it.id.upper()] = (
                            tok.upper().replace('/', ''), '(positional)'
                        )
                break

    # Post-parse clamp: 0 is invalid on Likert (1-5) rubrics
    results = []
    for it in rubric_items:
        raw_score, rationale = found.get(it.id.upper(), ('NA', '(no score found)'))
        is_binary = (it.scale or '').upper() == 'BINARY'
        if raw_score == '0' and not is_binary:
            raw_score = 'NA'
            rationale = '(out-of-range-clamped)'
        results.append(JudgeScore(
            id=new_id('sc'),
            judge_id=judge_id,
            rubric_item_id=it.id,
            score=raw_score,
            rationale=rationale,
        ))
    return results


def parse_single_item(
    raw: str,
    rubric_item: RubricItem,
    judge_id: str,
) -> JudgeScore:
    """Lenient parser for per-item retry responses."""
    rubric_binary = (rubric_item.scale or '').upper() == 'BINARY'
    rid  = rubric_item.id
    text = raw.strip()

    # 1) Full multi-strategy parser
    scores = parse_pipe_format(text, [rubric_item], judge_id)
    s = scores[0]
    if str(s.score).upper() not in ('', 'NA', 'N/A', 'NONE'):
        return s

    # 2) Bare leading digit / NA
    m = re.match(r'\s*(NA|N/A|[0-5])\b', text, re.IGNORECASE)
    if m:
        tok = m.group(1).upper().replace('/', '')
        if tok == '0' and not rubric_binary:
            tok = 'NA'
        return JudgeScore(id=new_id('sc'), judge_id=judge_id,
                          rubric_item_id=rid, score=tok,
                          rationale='(item-leading)')

    # 3) Prose word anywhere in first 80 chars
    head = text[:80].lower()
    pool = PROSE_TO_BINARY if rubric_binary else PROSE_TO_LIKERT
    for word, val in sorted(pool.items(), key=lambda kv: -len(kv[0])):
        if word in head:
            return JudgeScore(id=new_id('sc'), judge_id=judge_id,
                              rubric_item_id=rid, score=val,
                              rationale=f'(item-prose: {word})')

    # 4) Any digit in first 80 chars
    m = re.search(r'\b([0-5])\b', head)
    if m:
        tok = m.group(1)
        if tok == '0' and not rubric_binary:
            tok = 'NA'
        return JudgeScore(id=new_id('sc'), judge_id=judge_id,
                          rubric_item_id=rid, score=tok,
                          rationale='(item-digit)')

    return JudgeScore(id=new_id('sc'), judge_id=judge_id,
                      rubric_item_id=rid, score='NA',
                      rationale='(no score)')


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _rubric_block(items: List[RubricItem], short: bool = False) -> str:
    out = []
    for it in items:
        scale = '1/0/NA' if (it.scale or '').upper() == 'BINARY' else '1-5'
        desc  = (it.description or '').split('.')[0].strip() if short \
                else (it.description or '').strip()
        out.append(f'  {it.id} [{scale}] {it.name}: {desc}')
    return '\n'.join(out)


def _scale_instr(items: List[RubricItem]) -> str:
    if all((it.scale or '').upper() == 'BINARY' for it in items):
        return '1 = meets criterion, 0 = does not meet, NA = not applicable'
    return '1 = poor, 2 = below average, 3 = average, 4 = good, 5 = excellent'


def _scale_instr_single(item: RubricItem) -> str:
    """One-line scale string for a single rubric item."""
    if (item.scale or '').upper() == 'BINARY':
        return '1=meets, 0=does not meet'
    return '1=poor 2=below-avg 3=avg 4=good 5=excellent'


def _abstract_example(items: List[RubricItem]) -> str:
    """Format example using ABSTRACT IDs (X1..XN) so the real rubric IDs
    (CE1..CE5, U1..U5, etc.) are NEVER present in the example text.
    """
    binary = all((it.scale or '').upper() == 'BINARY' for it in items)
    cycle  = [1, 0, 1, 0, 1] if binary else [4, 3, 5, 2, 4]
    lines  = []
    for i in range(len(items)):
        lines.append(f'X{i+1}: {cycle[i % len(cycle)]} | ok.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class BaseAdapter:
    MODEL_FAMILY: str   = 'generic'
    ENDPOINT: str       = ENDPOINT_COMPLETION
    MAX_NEW_TOKENS: int = 300
    STOP: List[str]     = []
    SKIP_BATCH: bool    = False

    ITEM_ENDPOINT: str  = ENDPOINT_COMPLETION

    def build_messages(self, rubric_items, question_text, answer_text):
        raise NotImplementedError

    def build_item_messages(self, rubric_item, question_text, answer_text):
        raise NotImplementedError

    def build_split_item_messages(self, rubric_item: RubricItem,
                                   question_text: str, answer_text: str) -> List[Dict]:
        """Absolute-minimum single-item prompt for the hard-split pass.

        Contains ONLY: scale, criterion name+description (first 100 chars),
        Q (first 150 chars), A (first 150 chars), and a primed ID prefix.
        Used when both batch and per-item retries still leave NA scores.
        Subclasses may override for model-specific formatting.
        """
        scale = _scale_instr_single(rubric_item)
        crit  = (rubric_item.description or '')[:100].strip()
        prompt = (
            f'Scale: {scale}.\n'
            f'Criterion {rubric_item.id} ({rubric_item.name}): {crit}\n'
            f'Q: {question_text[:150]}\n'
            f'A: {answer_text[:150]}\n'
            f'{rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def extra_params(self) -> Dict:
        p: Dict = {'max_tokens': self.MAX_NEW_TOKENS}
        if self.STOP:
            p['stop'] = list(self.STOP)
        return p

    def extra_params_item(self) -> Dict:
        p: Dict = {'max_tokens': 64}
        if self.STOP:
            p['stop'] = list(self.STOP)
        return p

    def extra_params_split(self) -> Dict:
        """Minimal token budget for the hard-split pass."""
        stop = list(self.STOP) if self.STOP else ['\n', '</s>']
        return {'max_tokens': 16, 'stop': stop}

    def parse(self, raw, rubric_items, judge_id):
        return parse_pipe_format(raw, rubric_items, judge_id)

    def parse_item(self, raw, rubric_item, judge_id):
        return parse_single_item(raw, rubric_item, judge_id)


# ---------------------------------------------------------------------------
# MedGemma
# ---------------------------------------------------------------------------

class MedGemmaAdapter(BaseAdapter):
    MODEL_FAMILY   = 'medgemma'
    ENDPOINT       = ENDPOINT_COMPLETION
    ITEM_ENDPOINT  = ENDPOINT_CHAT
    MAX_NEW_TOKENS = 350
    STOP           = ['\nUSER:', '\nSYSTEM:', '\nASSISTANT:', '\nFINAL', '\n```', '\n\n']

    def build_messages(self, rubric_items, question_text, answer_text):
        system = (
            'You are a strict medical evaluator. Score each rubric item below '
            'using ONLY the format shown. Do not add any other text.\n\n'
            'SCORING SCALE: ' + _scale_instr(rubric_items) + '\n\n'
            'RUBRIC ITEMS:\n' + _rubric_block(rubric_items) + '\n\n'
            'REQUIRED OUTPUT — one line per rubric item, EXACT format:\n'
            'ID: SCORE | one-line rationale\n'
            'Output ONLY the score lines. Do not write FINAL ANSWER or any summary.\n\n'
            'FORMAT EXAMPLE (abstract IDs — do not copy scores):\n'
            + _abstract_example(rubric_items)
        )
        user = (
            f'QUESTION: {question_text}\n\n'
            f'ANSWER TO SCORE: {answer_text}\n\n'
            f'Produce exactly {len(rubric_items)} score lines now:'
        )
        return [{'prompt': f'SYSTEM: {system}\nUSER: {user}\nASSISTANT:'}]

    def build_item_messages(self, rubric_item, question_text, answer_text):
        scale = ('1 (meets) / 0 (does not meet) / NA'
                 if (rubric_item.scale or '').upper() == 'BINARY'
                 else '1-5 (1 worst, 5 best)')
        system = (
            'You are a strict medical evaluator. Score ONE rubric item.\n'
            f'Reply with EXACTLY one line: {rubric_item.id}: <SCORE> | <rationale>\n'
            f'SCORE must be {scale}. No other text.'
        )
        user = (
            f'QUESTION: {question_text}\n\nANSWER: {answer_text}\n\n'
            f'RUBRIC ITEM {rubric_item.id} — {rubric_item.name}\n'
            f'Criterion: {rubric_item.description}\n\n'
            f'Score for {rubric_item.id}:'
        )
        return [
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': user},
        ]

    def build_split_item_messages(self, rubric_item, question_text, answer_text):
        """Hard-split pass: minimal chat message for MedGemma."""
        scale = ('1=meets 0=does-not-meet'
                 if (rubric_item.scale or '').upper() == 'BINARY'
                 else '1-5')
        return [
            {'role': 'system', 'content': f'Score {rubric_item.id}. {scale}. Reply: {rubric_item.id}: SCORE'},
            {'role': 'user',   'content': (
                f'Q: {question_text[:150]}\n'
                f'A: {answer_text[:150]}\n'
                f'Item: {rubric_item.name} — {(rubric_item.description or "")[:80]}'
            )},
        ]

    def extra_params_item(self) -> Dict:
        return {'max_tokens': 80, 'stop': list(self.STOP)}

    def extra_params_split(self) -> Dict:
        return {'max_tokens': 16, 'stop': ['\n', '\nUSER:', '\nSYSTEM:']}


# ---------------------------------------------------------------------------
# BioMistral-7B — per-item only (SKIP_BATCH=True)
# ---------------------------------------------------------------------------

class BioMistralDAREAdapter(BaseAdapter):
    MODEL_FAMILY   = 'biomistral_dare'
    ENDPOINT       = ENDPOINT_COMPLETION
    ITEM_ENDPOINT  = ENDPOINT_COMPLETION
    MAX_NEW_TOKENS = 16
    STOP           = ['</s>', '\n\n', '\n']
    SKIP_BATCH     = True

    def build_messages(self, rubric_items, question_text, answer_text):
        """Not used (SKIP_BATCH=True) — kept for unit tests."""
        n       = len(rubric_items)
        ids_str = ' '.join(it.id for it in rubric_items)
        binary  = all((it.scale or '').upper() == 'BINARY' for it in rubric_items)
        if binary:
            example = ' '.join('1' if i % 2 == 0 else '0' for i in range(n))
            scale   = '1=meets 0=does-not-meet'
        else:
            cycle   = [3, 4, 5, 3, 4]
            example = ' '.join(str(cycle[i % len(cycle)]) for i in range(n))
            scale   = '1=poor 2=below-avg 3=avg 4=good 5=excellent'
        prompt = (
            f'Task: score {n} items ({ids_str}) using scale [{scale}].\n'
            f'Output ONLY {n} space-separated integers, e.g.: {example}\n'
            f'Q: {question_text[:180]}\n'
            f'A: {answer_text[:180]}\n'
            f'Scores:'
        )
        return [{'prompt': prompt}]

    def build_item_messages(self, rubric_item, question_text, answer_text):
        binary = (rubric_item.scale or '').upper() == 'BINARY'
        scale  = '1=meets, 0=does-not-meet' if binary else '1=poor, 2=below-avg, 3=avg, 4=good, 5=excellent'
        eg     = '1' if binary else '4'
        crit   = (rubric_item.description or '')[:80].strip()
        prompt = (
            f'Score rubric item {rubric_item.id}: {rubric_item.name}.\n'
            f'Criterion: {crit}\n'
            f'Scale: {scale}. Output ONE integer only, e.g. {eg}\n'
            f'Q: {question_text[:120]}\n'
            f'A: {answer_text[:120]}\n'
            f'{rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def build_split_item_messages(self, rubric_item, question_text, answer_text):
        binary = (rubric_item.scale or '').upper() == 'BINARY'
        eg     = '1' if binary else '3'
        scale  = '1/0' if binary else '1-5'
        prompt = (
            f'{scale}. {rubric_item.name}: {(rubric_item.description or "")[:60]}\n'
            f'Q:{question_text[:100]} A:{answer_text[:100]}\n'
            f'e.g.{eg} {rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def extra_params(self) -> Dict:
        return {'max_tokens': self.MAX_NEW_TOKENS, 'stop': list(self.STOP)}

    def extra_params_item(self) -> Dict:
        return {'max_tokens': 8, 'stop': ['\n', '</s>']}

    def extra_params_split(self) -> Dict:
        return {'max_tokens': 8, 'stop': ['\n', '</s>', ' ']}


class BioMistralBaseAdapter(BioMistralDAREAdapter):
    MODEL_FAMILY = 'biomistral_base'


# ---------------------------------------------------------------------------
# MedAlpaca-7B
# ---------------------------------------------------------------------------

class MedAlpacaAdapter(BaseAdapter):
    MODEL_FAMILY   = 'medalpaca'
    ENDPOINT       = ENDPOINT_COMPLETION
    ITEM_ENDPOINT  = ENDPOINT_COMPLETION
    MAX_NEW_TOKENS = 128
    STOP           = ['[INST]', '</s>', '\n\n\n', '\n\n']

    def build_messages(self, rubric_items, question_text, answer_text):
        first_id = rubric_items[0].id
        prompt = (
            f'[INST] You are a medical evaluator. Score every rubric item.\n'
            f'SCALE: {_scale_instr(rubric_items)}\n'
            f'FORMAT: one line per item — ID: SCORE | reason\n'
            f'FORMAT EXAMPLE (abstract IDs, do not copy these scores):\n'
            f'{_abstract_example(rubric_items)}\n\n'
            f'RUBRIC:\n{_rubric_block(rubric_items, short=True)}\n\n'
            f'QUESTION: {question_text[:200]}\n'
            f'ANSWER: {answer_text[:200]}\n'
            f'Score all {len(rubric_items)} items: [/INST]\n'
            f'{first_id}: '
        )
        return [{'prompt': prompt}]

    def build_item_messages(self, rubric_item, question_text, answer_text):
        scale = ('1 (meets) / 0 (does not meet)'
                 if (rubric_item.scale or '').upper() == 'BINARY' else '1-5')
        prompt = (
            f'[INST] Score ONE item. Reply: {rubric_item.id}: SCORE | reason. '
            f'SCORE={scale}.\n'
            f'Item: {rubric_item.name} — {rubric_item.description[:100]}\n'
            f'Q: {question_text[:150]}\nA: {answer_text[:150]} [/INST]\n'
            f'{rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def build_split_item_messages(self, rubric_item, question_text, answer_text):
        scale = '1/0' if (rubric_item.scale or '').upper() == 'BINARY' else '1-5'
        prompt = (
            f'[INST] {rubric_item.id} {rubric_item.name} {scale}. '
            f'Q:{question_text[:120]} A:{answer_text[:120]} [/INST]\n'
            f'{rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def extra_params_split(self) -> Dict:
        return {'max_tokens': 16, 'stop': ['[INST]', '</s>', '\n']}


# ---------------------------------------------------------------------------
# Meditron-7B
# ---------------------------------------------------------------------------

class MeditronAdapter(BaseAdapter):
    MODEL_FAMILY   = 'meditron'
    ENDPOINT       = ENDPOINT_COMPLETION
    ITEM_ENDPOINT  = ENDPOINT_COMPLETION
    MAX_NEW_TOKENS = 300
    STOP           = ['\n\n\n', '###', 'Question:', '[INST]', 'FORMAT']

    @staticmethod
    def _numeric_example_abstract(items: List[RubricItem]) -> str:
        binary = all((it.scale or '').upper() == 'BINARY' for it in items)
        cycle  = [1, 0, 1, 0, 1] if binary else [4, 3, 5, 2, 4]
        return '\n'.join(
            f'X{i+1}: {cycle[i % len(cycle)]}'
            for i in range(len(items))
        )

    def build_messages(self, rubric_items, question_text, answer_text):
        first_id  = rubric_items[0].id
        scale_str = _scale_instr(rubric_items)
        example   = self._numeric_example_abstract(rubric_items)
        rubric_lines = '\n'.join(
            f'{it.id}: {it.name}'
            for it in rubric_items
        )
        prompt = (
            f'FORMAT (do not copy these scores):\n'
            f'{example}\n'
            f'Scale: {scale_str}\n'
            f'Score the following answer on each rubric item. '
            f'Output one line per item: ID: SCORE\n'
            f'Rubric:\n{rubric_lines}\n'
            f'Q: {question_text[:300]}\n'
            f'A: {answer_text[:300]}\n'
            f'Scores:\n'
            f'{first_id}: '
        )
        return [{'prompt': prompt}]

    def build_item_messages(self, rubric_item, question_text, answer_text):
        scale = ('binary: 1=meets, 0=does not meet'
                 if (rubric_item.scale or '').upper() == 'BINARY'
                 else 'likert 1 (poor) to 5 (excellent)')
        prompt = (
            f'Score ONE item. {scale}.\n'
            f'Format: {rubric_item.id}: SCORE\n'
            f'Q: {question_text[:200]}\n'
            f'A: {answer_text[:200]}\n'
            f'Item {rubric_item.id} ({rubric_item.name}): '
            f'{rubric_item.description[:120]}\n'
            f'{rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def build_split_item_messages(self, rubric_item, question_text, answer_text):
        scale = '1/0' if (rubric_item.scale or '').upper() == 'BINARY' else '1-5'
        prompt = (
            f'{rubric_item.id} {rubric_item.name} [{scale}]\n'
            f'{(rubric_item.description or "")[:80]}\n'
            f'Q:{question_text[:120]} A:{answer_text[:120]}\n'
            f'{rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def extra_params_split(self) -> Dict:
        return {'max_tokens': 16, 'stop': ['\n\n', '###', 'Q:', 'FORMAT']}


# ---------------------------------------------------------------------------
# AdaptLLM/medicine-chat
# ---------------------------------------------------------------------------

class MedicineChatAdapter(BaseAdapter):
    MODEL_FAMILY   = 'medicine_chat'
    ENDPOINT       = ENDPOINT_COMPLETION
    ITEM_ENDPOINT  = ENDPOINT_COMPLETION
    MAX_NEW_TOKENS = 256
    STOP           = ['\n\n\n', '</s>', '[INST]', '### Instruction']

    def build_messages(self, rubric_items, question_text, answer_text):
        prompt = (
            '[INST] <<SYS>>\n'
            'You are a strict medical evaluator. Score every rubric item using '
            'ONLY the format shown.\n'
            'SCALE: ' + _scale_instr(rubric_items) + '\n'
            'OUTPUT: one line per item, format "ID: SCORE | reason".\n'
            '<</SYS>>\n\n'
            'RUBRIC ITEMS:\n' + _rubric_block(rubric_items, short=True) + '\n\n'
            'FORMAT EXAMPLE (abstract IDs — do not copy scores):\n'
            + _abstract_example(rubric_items) + '\n\n'
            f'QUESTION: {question_text}\n'
            f'ANSWER: {answer_text}\n\n'
            f'Now produce exactly {len(rubric_items)} score lines:\n[/INST]\n'
        )
        return [{'prompt': prompt}]

    def build_item_messages(self, rubric_item, question_text, answer_text):
        scale = ('1 (meets) / 0 (does not meet) / NA'
                 if (rubric_item.scale or '').upper() == 'BINARY' else '1-5')
        prompt = (
            '[INST] Score ONE rubric item.\n'
            f'Reply with EXACTLY: {rubric_item.id}: <SCORE> | <one-line reason>\n'
            f'SCORE must be {scale}.\n\n'
            f'QUESTION: {question_text}\nANSWER: {answer_text}\n\n'
            f'RUBRIC ITEM {rubric_item.id} — {rubric_item.name}: '
            f'{rubric_item.description}\n[/INST]\n'
            f'{rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def build_split_item_messages(self, rubric_item, question_text, answer_text):
        scale = '1/0' if (rubric_item.scale or '').upper() == 'BINARY' else '1-5'
        prompt = (
            f'[INST] <<SYS>>Score {rubric_item.id} [{scale}].<</SYS>>\n'
            f'{rubric_item.name}: {(rubric_item.description or "")[:80]}\n'
            f'Q:{question_text[:120]} A:{answer_text[:120]}[/INST]\n'
            f'{rubric_item.id}: '
        )
        return [{'prompt': prompt}]

    def extra_params_split(self) -> Dict:
        return {'max_tokens': 16, 'stop': ['</s>', '[INST]', '\n\n']}


# ---------------------------------------------------------------------------
# BioMedLM
# ---------------------------------------------------------------------------

class BioMedLMAdapter(BaseAdapter):
    MODEL_FAMILY   = 'biomedlm'
    ENDPOINT       = ENDPOINT_COMPLETION
    ITEM_ENDPOINT  = ENDPOINT_COMPLETION
    MAX_NEW_TOKENS = 96
    STOP           = ['\n\n', 'Q:', 'Question:']

    def build_messages(self, rubric_items, question_text, answer_text):
        q = question_text[:160]
        a = answer_text[:160]
        items = ', '.join(f'{it.id}({it.name.split()[0]})' for it in rubric_items)
        return [{'prompt': (
            f'Q: {q}\nA: {a}\n'
            f'Rate {items}. {_scale_instr(rubric_items)}.\n'
            f'{rubric_items[0].id}: '
        )}]

    def build_item_messages(self, rubric_item, question_text, answer_text):
        scale = '1 or 0' if (rubric_item.scale or '').upper() == 'BINARY' else '1-5'
        return [{'prompt': (
            f'Q: {question_text[:120]}\nA: {answer_text[:120]}\n'
            f'Rate {rubric_item.id} ({rubric_item.name}). {scale}.\n'
            f'{rubric_item.id}: '
        )}]

    def build_split_item_messages(self, rubric_item, question_text, answer_text):
        scale = '0/1' if (rubric_item.scale or '').upper() == 'BINARY' else '1-5'
        return [{'prompt': (
            f'Q:{question_text[:100]} A:{answer_text[:100]}\n'
            f'{rubric_item.id}[{scale}]: '
        )}]

    def extra_params_split(self) -> Dict:
        return {'max_tokens': 8, 'stop': ['\n', 'Q:', ' ']}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[str, BaseAdapter] = {
    'medgemma':        MedGemmaAdapter(),
    'biomistral':      BioMistralDAREAdapter(),
    'biomistral_base': BioMistralBaseAdapter(),
    'medicine_chat':   MedicineChatAdapter(),
    'meditron':        MeditronAdapter(),
    'medalpaca':       MedAlpacaAdapter(),
    'biomedlm':        BioMedLMAdapter(),
}


def get_adapter(judge_id: str) -> BaseAdapter:
    """Return the adapter for a judge id, falling back to BioMistralDAREAdapter."""
    return ADAPTER_REGISTRY.get(judge_id, BioMistralDAREAdapter())
