#!/usr/bin/env python3
"""
build_source_datasets.py

Fetches 3 source medical QA datasets from committed CSVs in the sibling repo:
  github.com/m22oct2000/Multi-LLMs-as-Judge/tree/main/dataset

Uses the GitHub Contents API (authenticated via GITHUB_TOKEN in .env) so it
works even when the sibling repo is private.  Falls back to the med_qa/US
split files (already in this same repo's dataset folder) if the sibling repo
is unavailable.

Output -> benchmark_dataset/source_datasets/
  health_advice.csv   100 rows  [question, answer, source]
  medquad.csv         100 rows  [question, answer, source]
  meddialog.csv       100 rows  [question, answer, source]
  curated_seed.csv    all rows  [question, answer, source]  (22 expert Q&A)

Usage:
  python benchmark_dataset/build_source_datasets.py
  python benchmark_dataset/build_source_datasets.py --n 200
"""

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import pandas as pd
import urllib.request
import urllib.error
import json

# ── Config ────────────────────────────────────────────────────────────────
SIBLING_OWNER = "m22oct2000"
SIBLING_REPO  = "Multi-LLMs-as-Judge"
SIBLING_REF   = "main"
SAMPLE_N      = 100

SOURCE_DATASETS = [
    {
        "filename":     "health_advice.csv",
        "api_path":     "dataset/medical_meadow_health_advice/medical_meadow_health_advice.csv",
        "rename":       {"instruction": "question", "output": "answer",
                         "input": "question"},   # MedAlpaca format
        "source_label": "medical_meadow_health_advice",
    },
    {
        "filename":     "medquad.csv",
        "api_path":     "dataset/MedQuAD/train.csv",
        "rename":       {"Question": "question", "Answer": "answer"},
        "source_label": "MedQuAD",
    },
    {
        "filename":     "meddialog.csv",
        "api_path":     "dataset/MedDialog/validation.csv",
        "rename":       {
            "input": "question",    "output": "answer",
            "utterance": "question","response": "answer",
            "patient": "question",  "doctor": "answer",
        },
        "source_label": "MedDialog",
    },
]

CURATED_SEED = {
    "filename":     "curated_seed.csv",
    "api_path":     "dataset/sample.csv",
    "source_label": "curated_expert",
}


# ── GitHub API helper ─────────────────────────────────────────────────────
def _github_token() -> str | None:
    """Read GITHUB_TOKEN from env or .env file."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    # try loading .env manually (dotenv may not be installed yet)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def fetch_csv_via_api(api_path: str, token: str | None, n_rows: int | None = None) -> pd.DataFrame:
    """
    Fetch a CSV from the sibling repo using the GitHub Contents API.
    For large files (>1 MB) the API returns a download_url; we follow it.
    Token is required for private repos.
    """
    api_url = (
        f"https://api.github.com/repos/{SIBLING_OWNER}/{SIBLING_REPO}"
        f"/contents/{api_path}?ref={SIBLING_REF}"
    )
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"  API → {api_path} ...", end=" ", flush=True)

    req = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        meta = json.loads(resp.read())

    # Large files: follow download_url (also needs auth for private repos)
    if meta.get("encoding") == "none" or "download_url" not in meta or meta.get("size", 0) > 500_000:
        dl_url = meta["download_url"]
        dl_req = urllib.request.Request(dl_url, headers={"Authorization": f"Bearer {token}"} if token else {})
        with urllib.request.urlopen(dl_req, timeout=120) as resp:
            content = resp.read().decode("utf-8", errors="replace")
    else:
        # Small file: base64-encoded inline
        content = base64.b64decode(meta["content"]).decode("utf-8", errors="replace")

    df = pd.read_csv(io.StringIO(content))
    if n_rows is not None:
        df = df.head(n_rows)
    print(f"{len(df)} rows, cols={list(df.columns)}")
    return df


# ── Column normalisation ───────────────────────────────────────────────────
def normalise(df: pd.DataFrame, rename: dict, source_label: str) -> pd.DataFrame:
    df = df.copy()
    actual_rename = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    # last-resort: map first two text columns to question/answer
    if "question" not in df.columns:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if text_cols:
            df = df.rename(columns={text_cols[0]: "question"})
            if len(text_cols) > 1 and "answer" not in df.columns:
                df = df.rename(columns={text_cols[1]: "answer"})

    missing = [c for c in ["question", "answer"] if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot map to [question, answer]. Missing: {missing}. Cols: {list(df.columns)}")

    df = df[["question", "answer"]].copy()
    df["source"] = source_label
    df.dropna(subset=["question", "answer"], inplace=True)
    df = df[df["question"].str.strip().str.len() > 0]
    df = df[df["answer"].str.strip().str.len() > 0]
    df.reset_index(drop=True, inplace=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────
def main(sample_n: int = SAMPLE_N) -> None:
    root     = Path(__file__).parent.parent
    dest_dir = root / "benchmark_dataset" / "source_datasets"
    dest_dir.mkdir(parents=True, exist_ok=True)

    token = _github_token()

    print(f"\n{'='*60}")
    print(f"build_source_datasets.py  (SAMPLE_N={sample_n})")
    print(f"Sibling repo : github.com/{SIBLING_OWNER}/{SIBLING_REPO}")
    print(f"GitHub token : {'✅ found' if token else '⚠️  NOT FOUND — will fail for private repo'}")
    print(f"Output dir   : {dest_dir}")
    print(f"{'='*60}\n")

    results = []

    # ── 3 main source datasets ───────────────────────────────────────────
    for ds in SOURCE_DATASETS:
        dest = dest_dir / ds["filename"]
        if dest.exists():
            n = len(pd.read_csv(dest))
            print(f"✅ {ds['filename']:30s} already exists ({n} rows) — skipping")
            results.append((ds["filename"], n, True))
            continue

        print(f"⬇️  {ds['filename']}")
        try:
            raw = fetch_csv_via_api(ds["api_path"], token, n_rows=sample_n)
            df  = normalise(raw, ds["rename"], ds["source_label"])
            df.to_csv(dest, index=False)
            print(f"   ✅ Saved {len(df)} rows → {dest.name}")
            results.append((ds["filename"], len(df), True))
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append((ds["filename"], 0, False))

    # ── curated seed ─────────────────────────────────────────────────────
    dest = dest_dir / CURATED_SEED["filename"]
    if dest.exists():
        n = len(pd.read_csv(dest))
        print(f"✅ {CURATED_SEED['filename']:30s} already exists ({n} rows) — skipping")
        results.append((CURATED_SEED["filename"], n, True))
    else:
        print(f"⬇️  {CURATED_SEED['filename']} (expert curated rows)")
        try:
            raw = fetch_csv_via_api(CURATED_SEED["api_path"], token)
            if "question" in raw.columns and "answer" in raw.columns:
                raw = raw[["question", "answer"]].copy()
                raw["source"] = CURATED_SEED["source_label"]
                raw.dropna(subset=["question", "answer"], inplace=True)
            else:
                raw = normalise(raw, {}, CURATED_SEED["source_label"])
            raw.to_csv(dest, index=False)
            print(f"   ✅ Saved {len(raw)} rows → {dest.name}")
            results.append((CURATED_SEED["filename"], len(raw), True))
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append((CURATED_SEED["filename"], 0, False))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("=== Source dataset summary ===")
    total   = 0
    all_ok  = True
    for fname, n, ok in results:
        status = "✅" if ok and n > 0 else "❌"
        print(f"  {status} {fname:35s} {n:>4} rows")
        if ok and n > 0:
            total += n
        else:
            all_ok = False
    print(f"\n  Total rows available : {total}")
    print(f"  Ready for build_agreement_dataset.py: {'YES ✅' if all_ok else 'NO ❌ — fix errors above'}")
    print(f"{'='*60}\n")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=SAMPLE_N,
                        help=f"Rows per dataset (default {SAMPLE_N})")
    args = parser.parse_args()
    main(sample_n=args.n)
