"""Evaluate candidate medical LLMs for the clinical-QA judge panel.

For each candidate model:
  1. Try to load it locally (bf16, CPU-friendly).
  2. If too big to load, fall back to HF serverless inference.
  3. Send a standardized rubric-scoring prompt (PEMAT 5 items, binary).
  4. Grade the response on format adherence, parseability, latency, sanity.

Outputs JSON to results/judge_candidate_eval.json so the notebook
and the final config can pick the top-4 panel automatically.

Run a single candidate (one subprocess per model is intentional so
CPU memory is freed cleanly between models):

    python tools/evaluate_judge_candidates.py <model_id> [--api]

Run the full sweep:

    bash tools/evaluate_all_candidates.sh
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ----------------------------------------------------------------------------
# Standardized PEMAT rubric prompt (binary 0/1 scoring, 5 items)
# ----------------------------------------------------------------------------

CLINICAL_QUESTION = (
    "What is the first-line treatment for an acute STEMI in an adult patient "
    "who arrives within the first 90 minutes of chest pain onset?"
)

CLINICAL_ANSWER = (
    "Activate the STEMI protocol immediately. Give aspirin 162-325 mg chewed. "
    "Call the cath lab for primary PCI within 90 minutes of first medical contact. "
    "Place the patient on cardiac monitoring with continuous ECG, give supplemental "
    "oxygen only if SpO2 < 90%, and obtain serial troponin and 12-lead ECG. "
    "If primary PCI is not available within 120 minutes, give a fibrinolytic agent."
)

RUBRIC_ITEMS: List[Dict[str, str]] = [
    {"id": "U1", "name": "Uses plain, everyday language",
     "desc": "Words a caregiver uses; jargon defined."},
    {"id": "U2", "name": "Information is organized logically",
     "desc": "Most important info first; ordered correctly."},
    {"id": "U3", "name": "Focuses only on key messages without clutter",
     "desc": "On topic; <=5 key points; no clutter."},
    {"id": "A1", "name": "Provides specific, actionable steps",
     "desc": "Tells caregiver exactly what to DO."},
    {"id": "A2", "name": "Addresses barriers to action",
     "desc": "Anticipates obstacles; offers workarounds."},
]

PROMPT_SYSTEM = (
    "You are a strict medical evaluator. Score each rubric item below on a "
    "BINARY scale: 1 = meets criterion, 0 = does not meet, NA = not applicable.\n"
    "RULES:\n"
    "  * Output EXACTLY one line per item, in this format and nothing else:\n"
    "        ID: SCORE | one-line rationale\n"
    "  * SCORE must be 1, 0, or NA.\n"
    "  * Do not add explanations outside the lines.\n\n"
    "RUBRIC ITEMS:\n"
    + "\n".join(f"  {it['id']}: {it['name']} -- {it['desc']}" for it in RUBRIC_ITEMS)
    + "\n\nFORMAT EXAMPLE:\n"
    "U1: 1 | Plain English throughout.\n"
    "U2: 0 | Steps presented out of order.\n"
    "U3: 1 | Five key points, no clutter.\n"
    "A1: 1 | Concrete action steps given.\n"
    "A2: 0 | No barriers addressed."
)

PROMPT_USER = (
    f"QUESTION: {CLINICAL_QUESTION}\n\n"
    f"ANSWER: {CLINICAL_ANSWER}\n\n"
    "Score every item now (5 lines total):"
)


def build_chat_messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user",   "content": PROMPT_USER},
    ]


def build_raw_prompt() -> str:
    return f"{PROMPT_SYSTEM}\n\n{PROMPT_USER}\n\n"


# ----------------------------------------------------------------------------
# Permissive parser used by the grader (we are also testing this design)
# ----------------------------------------------------------------------------

ITEM_IDS = [it["id"] for it in RUBRIC_ITEMS]

PROSE_TO_BINARY = {
    "meets": 1, "yes": 1, "present": 1, "y": 1, "true": 1, "agree": 1,
    "does not meet": 0, "not meet": 0, "no": 0, "absent": 0, "n": 0,
    "false": 0, "disagree": 0, "fail": 0, "fails": 0,
}

RE_PIPE      = re.compile(r"\b([A-Z]{1,3}\d{0,2})\s*[:\-=]\s*([01]|NA)\s*\|", re.IGNORECASE)
RE_PAREN     = re.compile(r"\b([A-Z]{1,3}\d{0,2})\s*[:\-]?\s*\(?([01])\)?", re.IGNORECASE)
RE_ID_PROSE  = re.compile(
    r"\b([A-Z]{1,3}\d{0,2})\s*[:\-]\s*(meets|does not meet|not meet|yes|no|present|absent|true|false|agree|disagree)",
    re.IGNORECASE,
)
RE_NUMBERED  = re.compile(r"^\s*\d+[\.\)]\s*([A-Z]{1,3}\d{0,2})\b", re.IGNORECASE | re.MULTILINE)
RE_BARE_NUM  = re.compile(r"\b([A-Z]{1,3}\d{0,2})\b[^\n]{0,40}?\b([01])\b", re.IGNORECASE)


def parse_scores(raw: str) -> Dict[str, str]:
    found: Dict[str, str] = {}

    # 1. Pipe format
    for m in RE_PIPE.finditer(raw):
        iid = m.group(1).upper()
        if iid in ITEM_IDS and iid not in found:
            found[iid] = str(m.group(2)).upper()

    # 2. "ID: meets / does not meet"
    for m in RE_ID_PROSE.finditer(raw):
        iid  = m.group(1).upper()
        word = m.group(2).lower().strip()
        if iid in ITEM_IDS and iid not in found:
            mapped = PROSE_TO_BINARY.get(word)
            if mapped is not None:
                found[iid] = str(mapped)

    # 3. "ID: (1)" / "ID 1" close pairing
    for m in RE_PAREN.finditer(raw):
        iid = m.group(1).upper()
        if iid in ITEM_IDS and iid not in found:
            found[iid] = m.group(2)

    # 4. Last resort: bare digit near ID anywhere on the line
    for m in RE_BARE_NUM.finditer(raw):
        iid = m.group(1).upper()
        if iid in ITEM_IDS and iid not in found:
            found[iid] = m.group(2)

    return found


EXAMPLE_RATIONALES = {
    "Plain English throughout.",
    "Steps presented out of order.",
    "Five key points, no clutter.",
    "Concrete action steps given.",
    "No barriers addressed.",
}


def grade_response(raw: str) -> Dict[str, Any]:
    parsed = parse_scores(raw)
    n_parsed = len(parsed)
    format_pct = n_parsed / len(ITEM_IDS) * 100.0

    # Detect verbatim copy of the example we put in the prompt
    overlap = sum(1 for e in EXAMPLE_RATIONALES if e in raw)
    echoed_example = overlap >= 3

    # Does the model actually reference the clinical content of the answer?
    clinical_terms = (
        "stemi", "aspirin", "pci", "cath", "ecg", "troponin", "oxygen",
        "fibrinolytic", "cardiac", "infarction", "rest",
    )
    sane = any(w in raw.lower() for w in clinical_terms)

    return {
        "parsed_items":     parsed,
        "n_parsed":         n_parsed,
        "format_pct":       round(format_pct, 1),
        "echoed_example":   echoed_example,
        "example_overlap":  overlap,
        "sane_content":     sane,
        "sample_raw":       raw[:500],
        "raw_chars":        len(raw),
    }


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def _safe_torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.float32)


def evaluate_local(model_id: str, dtype: str = "bfloat16",
                   max_new: int = 192) -> Dict[str, Any]:
    """Load the model locally with transformers and run the prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{model_id}] loading locally (dtype={dtype}) ...", flush=True)
    t0 = time.time()
    tok_kw = {}
    if os.environ.get("TRUST_REMOTE_CODE") == "1":
        tok_kw["trust_remote_code"] = True
    tok = AutoTokenizer.from_pretrained(
        model_id, cache_dir=os.environ.get("HF_HOME"),
        token=os.environ.get("HF_TOKEN"),
        **tok_kw,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    extra_kw = {}
    if os.environ.get("TRUST_REMOTE_CODE") == "1":
        extra_kw["trust_remote_code"] = True
    offload = os.environ.get("OFFLOAD_FOLDER")
    if offload:
        os.makedirs(offload, exist_ok=True)
        extra_kw.update({"device_map": "auto", "offload_folder": offload})
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=os.environ.get("HF_HOME"),
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=_safe_torch_dtype(dtype),
        low_cpu_mem_usage=True,
        **extra_kw,
    )
    model.eval()
    load_s = time.time() - t0
    print(f"[{model_id}] loaded in {load_s:.1f}s", flush=True)

    has_chat = bool(getattr(tok, "chat_template", None))
    endpoint_used = "raw_prompt"
    prompt_text = build_raw_prompt()
    if has_chat:
        try:
            prompt_text = tok.apply_chat_template(
                build_chat_messages(), tokenize=False, add_generation_prompt=True,
            )
            endpoint_used = "chat_template"
        except Exception:
            try:
                # Some templates (Mistral) don't allow a system role; merge it
                # into the first user message.
                msgs = build_chat_messages()
                merged = [{"role": "user",
                           "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}]
                prompt_text = tok.apply_chat_template(
                    merged, tokenize=False, add_generation_prompt=True,
                )
                endpoint_used = "chat_template_merged_system"
            except Exception:
                prompt_text = build_raw_prompt()
                endpoint_used = "raw_prompt_fallback"

    inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=2048)

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=False,
        )
    gen_s = time.time() - t1
    raw  = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)

    grade = grade_response(raw)

    del model, tok
    gc.collect()

    return {
        "model_id":      model_id,
        "test_path":     "local",
        "loaded":        True,
        "endpoint_used": endpoint_used,
        "has_chat_template": has_chat,
        "load_seconds":  round(load_s, 1),
        "gen_seconds":   round(gen_s, 1),
        "dtype":         dtype,
        **grade,
    }


def evaluate_hf_inference(model_id: str, max_new: int = 192) -> Dict[str, Any]:
    """Use the HF Inference Providers API (token required)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"model_id": model_id, "test_path": "hf_api", "loaded": False,
                "error": "HF_TOKEN not set"}

    # Try the chat-completions style first (works with text-generation models
    # served via huggingface_hub Inference Providers).
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type":  "application/json"}
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}/v1/chat/completions"
    payload = {
        "model":       model_id,
        "messages":    build_chat_messages(),
        "max_tokens":  max_new,
        "temperature": 0.0,
    }
    t0 = time.time()
    try:
        with httpx.Client(timeout=120.0) as c:
            r = c.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            # Try the text-generation endpoint as a fallback (raw prompt)
            tg_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
            tg_payload = {
                "inputs": build_raw_prompt(),
                "parameters": {
                    "max_new_tokens": max_new,
                    "temperature":    0.0,
                    "return_full_text": False,
                },
            }
            with httpx.Client(timeout=120.0) as c:
                r = c.post(tg_url, headers=headers, json=tg_payload)
            r.raise_for_status()
            data = r.json()
            raw = data[0]["generated_text"] if isinstance(data, list) else data.get(
                "generated_text", str(data))
            endpoint_used = "hf_textgen"
        else:
            data = r.json()
            raw = data["choices"][0]["message"]["content"]
            endpoint_used = "hf_chat"
    except Exception as e:
        return {"model_id": model_id, "test_path": "hf_api",
                "loaded": False, "error": str(e)[:300]}

    grade = grade_response(raw)
    return {
        "model_id":      model_id,
        "test_path":     "hf_api",
        "loaded":        True,
        "endpoint_used": endpoint_used,
        "load_seconds":  0.0,
        "gen_seconds":   round(time.time() - t0, 1),
        **grade,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--api",    action="store_true",
                    help="Use HF inference API instead of local load")
    ap.add_argument("--dtype",  default="bfloat16")
    ap.add_argument("--maxnew", type=int, default=192)
    ap.add_argument("--out",    default="results/judge_candidate_eval.jsonl")
    args = ap.parse_args()

    try:
        if args.api:
            res = evaluate_hf_inference(args.model_id, max_new=args.maxnew)
        else:
            res = evaluate_local(args.model_id, dtype=args.dtype, max_new=args.maxnew)
    except Exception as e:
        res = {"model_id": args.model_id,
               "test_path": "api" if args.api else "local",
               "loaded": False,
               "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-1200:]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")

    print(json.dumps({k: v for k, v in res.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
