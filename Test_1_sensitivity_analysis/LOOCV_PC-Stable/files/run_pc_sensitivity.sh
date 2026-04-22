#!/usr/bin/env bash
# =============================================================================
# run_pc_sensitivity.sh
#
# Runs all 7 PC-stable parameter combinations sequentially.
# Fold IDs are generated on the first call and reused by all subsequent calls.
#
# Usage:
#   chmod +x run_pc_sensitivity.sh
#   ./run_pc_sensitivity.sh <input_dir> 2>&1 | tee pc_sensitivity_master.log
# =============================================================================

set -euo pipefail

INPUT_DIR="${1:?Usage: ./run_pc_sensitivity.sh <input_dir>}"
SCRIPT="CBNs_PC_sensitivity.R"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: $SCRIPT not found in $(pwd)"
  exit 1
fi

run_combo() {
  local label="$1"; shift
  echo ""
  echo "========================================================"
  echo "  Starting combination: $label"
  echo "  $(date)"
  echo "========================================================"
  Rscript "$SCRIPT" "$INPUT_DIR" "$@"
  echo "  Finished: $label  ($(date))"
}

# | Config | alpha | max.sx | test | Interpretation                        |
# |--------|-------|--------|------|---------------------------------------|
# | 0      | 0.01  | NULL   | mi   | Original experiment (baseline)        |
# | 1      | 0.05  | 3      | mi   | Default setup                         |
# | 2      | 0.01  | 3      | mi   | Conservative CI (sparser graph)       |
# | 3      | 0.10  | 3      | mi   | Permissive CI (denser graph)          |
# | 4      | 0.05  | 2      | mi   | Limited conditioning                  |
# | 5      | 0.05  | 4      | mi   | Deeper conditioning                   |
# | 6      | 0.05  | NULL   | mi   | Full PC (no conditioning limit)       |
# | 7      | 0.05  | 3      | x2   | Alternative CI test                   |

run_combo "config_0_alpha0.01_maxsxNULL_mi_original" --alpha 0.01 --max.sx NULL --test mi  --fraction 0.25
run_combo "config_1_alpha0.05_maxsx3_mi"             --alpha 0.05 --max.sx 3    --test mi  --fraction 0.25
run_combo "config_2_alpha0.01_maxsx3_mi"             --alpha 0.01 --max.sx 3    --test mi  --fraction 0.25
run_combo "config_3_alpha0.10_maxsx3_mi"             --alpha 0.10 --max.sx 3    --test mi  --fraction 0.25
run_combo "config_4_alpha0.05_maxsx2_mi"             --alpha 0.05 --max.sx 2    --test mi  --fraction 0.25
run_combo "config_5_alpha0.05_maxsx4_mi"             --alpha 0.05 --max.sx 4    --test mi  --fraction 0.25
run_combo "config_6_alpha0.05_maxsxNULL_mi"          --alpha 0.05 --max.sx NULL --test mi  --fraction 0.25
run_combo "config_7_alpha0.05_maxsx3_x2"             --alpha 0.05 --max.sx 3    --test x2  --fraction 0.25

echo ""
echo "========================================================"
echo "  All combinations completed: $(date)"
echo "  Fold IDs: ./pc_sensitivity_folds/fold_ids.txt"
echo "  Fold detail: ./pc_sensitivity_folds/fold_ids_detail.csv"
echo "========================================================"
