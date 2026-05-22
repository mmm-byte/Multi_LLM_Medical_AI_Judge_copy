#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full pipeline runner for Multi-LLM-as-Judge Medical AI
#
# Steps:
#   0. Tests (no LLM needed)
#   1. Build agreement benchmark CSV
#   2. Build adrd_questions.json (for exp3) from the same CSV
#   3. Run Exp 1 — dataset analysis (no LLM)
#   4. Run Exp 2 — core agreement analysis (requires vLLM)
#   5. Run Exp 3 — rubric sensitivity (requires vLLM)
#   6. Run Exp 4 — box plot figures (reads exp2 results, no LLM)
#
# Prerequisites:
#   - Python 3.8+
#   - 4 vLLM servers running on ports 8001-8004
#   - Meditron: started with --chat-template ""
#
# Usage:
#   bash run_all.sh                        # full pipeline (synthetic benchmark)
#   bash run_all.sh --use-source-datasets  # full pipeline with real medical QA datasets
#                                          #   (MedQuAD / MedDialog / medical_meadow)
#                                          #   Requires CSVs in benchmark_dataset/source_datasets/
#                                          #   See benchmark_dataset/source_datasets/README.md
#   bash run_all.sh --no-llm               # steps 0-3 + step 6 only
#   bash run_all.sh --tests-only           # only run tests
#   bash run_all.sh --use-source-datasets --no-llm  # build source benchmark, skip LLM steps
# =============================================================================

set -euo pipefail

PYTHON=${PYTHON:-python3}
NO_LLM=false
TESTS_ONLY=false
USE_SRC=false

for arg in "$@"; do
  case $arg in
    --no-llm)              NO_LLM=true ;;
    --tests-only)          TESTS_ONLY=true ;;
    --use-source-datasets) USE_SRC=true ;;
  esac
done

# Export env var consumed by build_agreement_dataset.py and build_adrd_questions.py
if $USE_SRC; then
  export USE_SOURCE_DATASETS=1
  echo "[INFO] Source-dataset mode ON: will ingest MedQuAD / MedDialog / medical_meadow CSVs"
  echo "[INFO] CSVs must be present in benchmark_dataset/source_datasets/"
  echo "[INFO] See benchmark_dataset/source_datasets/README.md for download instructions"
else
  export USE_SOURCE_DATASETS=0
fi

echo "======================================================="
echo " Multi-LLM-as-Judge Medical AI — Full Pipeline"
echo "======================================================="
echo ""

# -----------------------------------------------------------
# Step 0: Tests
# -----------------------------------------------------------
echo "[STEP 0] Running test suite..."
$PYTHON tests/test_core.py
echo ""

if $TESTS_ONLY; then
  echo "--tests-only flag set. Done."
  exit 0
fi

# -----------------------------------------------------------
# Step 1: Build agreement benchmark CSV
# -----------------------------------------------------------
echo "[STEP 1] Building agreement benchmark CSV..."
$PYTHON benchmark_dataset/build_agreement_dataset.py
echo ""

# -----------------------------------------------------------
# Step 2: Build adrd_questions.json (required by exp3)
# -----------------------------------------------------------
echo "[STEP 2] Building adrd_questions.json for exp3..."
$PYTHON benchmark_dataset/build_adrd_questions.py
echo ""

# -----------------------------------------------------------
# Step 3: Exp 1 — dataset analysis (no LLM)
# -----------------------------------------------------------
echo "[STEP 3] Exp 1: Dataset analysis..."
$PYTHON experiments/exp1_dataset_analysis.py
echo ""

if $NO_LLM; then
  echo "--no-llm flag: skipping Exp 2 and Exp 3."
  echo ""
  echo "[STEP 6] Skipping box plots (no Exp 2 results yet)."
  echo "Run without --no-llm to get Exp 2 results, then:"
  echo "  $PYTHON experiments/exp4_boxplot_agreement.py"
  exit 0
fi

# -----------------------------------------------------------
# Step 4: Exp 2 — Core agreement analysis (requires vLLM)
# -----------------------------------------------------------
echo "[STEP 4] Exp 2: Per-rubric agreement analysis..."
$PYTHON experiments/exp2_agreement_analysis.py
echo ""

# -----------------------------------------------------------
# Step 5: Exp 3 — Rubric sensitivity (requires vLLM)
# -----------------------------------------------------------
echo "[STEP 5] Exp 3: Rubric sensitivity analysis..."
$PYTHON experiments/exp3_rubric_sensitivity.py
echo ""

# -----------------------------------------------------------
# Step 6: Exp 4 — Box plot figures (reads exp2 results)
# -----------------------------------------------------------
echo "[STEP 6] Exp 4: Generating box plot figures..."
$PYTHON experiments/exp4_boxplot_agreement.py
echo ""

echo "======================================================="
echo " Pipeline complete."
echo " Results : results/"
echo " Figures : results/figures/"
echo " CSV     : benchmark_dataset/agreement_benchmark.csv"
echo "======================================================="
