#!/bin/bash
# =============================================================================
# run_combo_H.sh
#
# Runs ONLY combination H (mi, alpha=0.05, max.sx=1) using the same
# CBNs_LOOCV_training_MMPC_sensitivity.R script and the same shared fold IDs
# as combinations A-G, then appends H to all_combinations_summary.csv.
#
# Follows exactly the same pattern as run_sensitivity_analysis.sh.
#
# Prerequisites:
#   - CBNs_LOOCV_training_MMPC_sensitivity.R in current directory
#   - sensitivity_results/fold_ids_used.txt must exist (from original A-G run)
#   - sensitivity_results/all_combinations_summary.csv must exist
#   - ./Shared_CSVs with the three input CSVs
#
# Usage:
#   chmod +x run_combo_H.sh
#   ./run_combo_H.sh 2>&1 | tee combo_H.log
# =============================================================================

SCRIPT="CBNs_LOOCV_training_MMPC_sensitivity.R"
INPUT_DIR="./Shared_CSVs"
REP="1"
PCT="01,50,100"
RESULTS_DIR="sensitivity_results"

# Combination H parameters
H_LABEL="H_mi_alpha0.05_maxsx1"
H_TEST="mi"
H_ALPHA="0.05"
H_MAXSX="1"
H_PURPOSE="Smaller conditioning set — near-marginal testing"

echo "=============================================================="
echo " Combination H: $H_TEST | alpha=$H_ALPHA | max.sx=$H_MAXSX"
echo " Purpose    : $H_PURPOSE"
echo " Script     : $SCRIPT"
echo " Started    : $(date)"
echo "=============================================================="

# --- Sanity checks ---
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: $SCRIPT not found in current directory."; exit 1
fi
if [ ! -f "${RESULTS_DIR}/fold_ids_used.txt" ]; then
  echo "ERROR: ${RESULTS_DIR}/fold_ids_used.txt not found."
  echo "Run run_sensitivity_analysis.sh first to generate shared fold IDs."
  exit 1
fi
if [ ! -f "${RESULTS_DIR}/all_combinations_summary.csv" ]; then
  echo "ERROR: ${RESULTS_DIR}/all_combinations_summary.csv not found."
  echo "Run run_sensitivity_analysis.sh first."
  exit 1
fi

# --- Read the shared fold IDs (same ones used by A-G) ---
FOLD_IDS=$(cat "${RESULTS_DIR}/fold_ids_used.txt")
echo "Shared fold IDs loaded: $(echo $FOLD_IDS | tr ',' '\n' | wc -l) folds"
echo "Fold IDs: $FOLD_IDS"
echo ""

# --- Run combination H ---
echo "--------------------------------------------------------------"
echo "Combination : $H_LABEL"
echo "Parameters  : test=$H_TEST  alpha=$H_ALPHA  max.sx=$H_MAXSX"
echo "Folds       : same shared IDs as A-G"
echo "Started     : $(date)"
echo "--------------------------------------------------------------"

LOG_FILE="${RESULTS_DIR}/${H_LABEL}.log"

COMBO_LABEL="$H_LABEL" Rscript "$SCRIPT" \
    "$INPUT_DIR" "$REP" "$PCT" \
    --alpha  "$H_ALPHA" \
    --max.sx "$H_MAXSX" \
    --test   "$H_TEST"  \
    --fold_ids "$FOLD_IDS" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "  COMPLETED : $(date)"
  grep "action_in_LC" "$LOG_FILE" | grep "Pct=" | tail -3 | sed 's/^/  /' 2>/dev/null || true
else
  echo "  FAILED (exit=$EXIT_CODE) : $(date)"
  echo "  Last 5 lines of log:"
  tail -5 "$LOG_FILE" | sed 's/^/    /'
  exit $EXIT_CODE
fi

# --- Append H to all_combinations_summary.csv ---
echo ""
echo "Appending combination H to ${RESULTS_DIR}/all_combinations_summary.csv..."

H_CSV="${RESULTS_DIR}/${H_LABEL}/global_summary.csv"

if [ ! -f "$H_CSV" ]; then
  echo "ERROR: $H_CSV not found — R script may have failed silently."
  exit 1
fi

# Append data rows only (skip header — it is already in the merged file)
tail -n +2 "$H_CSV" >> "${RESULTS_DIR}/all_combinations_summary.csv"

TOTAL=$(wc -l < "${RESULTS_DIR}/all_combinations_summary.csv")
H_ROWS=$(tail -n +2 "$H_CSV" | wc -l)
echo "Appended $H_ROWS rows from $H_CSV"
echo "Total lines in all_combinations_summary.csv now: $TOTAL (including header)"

echo ""
echo "=============================================================="
echo " DONE: $(date)"
echo " Combination H results : ${RESULTS_DIR}/${H_LABEL}/global_summary.csv"
echo " Merged CSV            : ${RESULTS_DIR}/all_combinations_summary.csv"
echo " Full log              : $LOG_FILE"
echo "=============================================================="
