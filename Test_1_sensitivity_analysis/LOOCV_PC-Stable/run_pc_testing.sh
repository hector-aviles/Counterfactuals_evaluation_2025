#!/usr/bin/env bash
# =============================================================================
# run_pc_testing.sh
#
# Runs counterfactual testing for all 7 PC-stable sensitivity combinations.
# Each combination is called independently — safe to rerun any one of them.
#
# Usage:
#   chmod +x run_pc_testing.sh
#   ./run_pc_testing.sh [--folds START-END] 2>&1 | tee pc_testing_master.log
#
# To resume a specific combination from fold 10:
#   python3 run_pc_testing.py ./pc_mi_alpha0p05_maxsx3_fracp25 --folds 10-78
# =============================================================================

set -euo pipefail

FOLDS="1-78"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folds) FOLDS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SCRIPT="run_pc_testing.py"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: $SCRIPT not found in $(pwd)"
  exit 1
fi

run_combo() {
  local combo_dir="$1"
  if [ ! -d "$combo_dir" ]; then
    echo "[Warning] Directory not found: $combo_dir — skipping"
    return
  fi
  echo ""
  echo "========================================================"
  echo "  Testing combination: $(basename $combo_dir)"
  echo "  Folds: $FOLDS"
  echo "  $(date)"
  echo "========================================================"
  python3 "$SCRIPT" "$combo_dir" --folds "$FOLDS"
  echo "  Finished: $(basename $combo_dir)  ($(date))"
}

# Label format: pc_<test>_alpha<alpha>_maxsx<max.sx>_frac<fraction>
run_combo "./pc_mi_alpha0p01_maxsxNULL_fracp25"   # Config 0 — original experiment
run_combo "./pc_mi_alpha0p05_maxsx3_fracp25"
run_combo "./pc_mi_alpha0p01_maxsx3_fracp25"
run_combo "./pc_mi_alpha0p10_maxsx3_fracp25"
run_combo "./pc_mi_alpha0p05_maxsx2_fracp25"
run_combo "./pc_mi_alpha0p05_maxsx4_fracp25"
run_combo "./pc_mi_alpha0p05_maxsxNULL_fracp25"
run_combo "./pc_x2_alpha0p05_maxsx3_fracp25"

echo ""
echo "========================================================"
echo "  All testing completed: $(date)"
echo "  Aggregate results in each combo dir:"
echo "    ./<combo>/twin_networks_results_all.csv"
echo "========================================================"
