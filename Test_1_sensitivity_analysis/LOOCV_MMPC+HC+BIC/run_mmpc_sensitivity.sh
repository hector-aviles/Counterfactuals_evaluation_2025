#!/usr/bin/env bash
# =============================================================================
# run_mmpc_sensitivity.sh
#
# Runs all 7 MMPC+HC+BIC parameter combinations sequentially.
# Fold IDs are generated on the first call and reused by all subsequent calls.
#
# Usage:
#   chmod +x run_mmpc_sensitivity.sh
#   ./run_mmpc_sensitivity.sh <input_dir> 2>&1 | tee mmpc_sensitivity_master.log
# =============================================================================

set -euo pipefail

INPUT_DIR="${1:?Usage: ./run_mmpc_sensitivity.sh <input_dir>}"
SCRIPT="CBNs_mmpc_sensitivity.R"

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

# | Config | alpha | max.sx | test | Purpose                      |
# |--------|-------|--------|------|------------------------------|
# | 1      | 0.05  | 3      | mi   | Default setup (baseline)     |
# | 2      | 0.01  | 3      | mi   | Conservative CI (fewer edges)|
# | 3      | 0.10  | 3      | mi   | Permissive CI (more edges)   |
# | 4      | 0.05  | 2      | mi   | Limited conditioning         |
# | 5      | 0.05  | 4      | mi   | Deeper conditioning          |
# | 6      | 0.01  | 2      | mi   | Conservative + shallow       |
# | 7      | 0.05  | 3      | x2   | Different CI test            |

run_combo "config_1_alpha0.05_maxsx3_mi"  --alpha 0.05 --max.sx 3 --test mi  --fraction 0.25
run_combo "config_2_alpha0.01_maxsx3_mi"  --alpha 0.01 --max.sx 3 --test mi  --fraction 0.25
run_combo "config_3_alpha0.10_maxsx3_mi"  --alpha 0.10 --max.sx 3 --test mi  --fraction 0.25
run_combo "config_4_alpha0.05_maxsx2_mi"  --alpha 0.05 --max.sx 2 --test mi  --fraction 0.25
run_combo "config_5_alpha0.05_maxsx4_mi"  --alpha 0.05 --max.sx 4 --test mi  --fraction 0.25
run_combo "config_6_alpha0.01_maxsx2_mi"  --alpha 0.01 --max.sx 2 --test mi  --fraction 0.25
run_combo "config_7_alpha0.05_maxsx3_x2"  --alpha 0.05 --max.sx 3 --test x2  --fraction 0.25

echo ""
echo "========================================================"
echo "  All combinations completed: $(date)"
echo "  Fold IDs: ./mmpc_sensitivity_folds/fold_ids.txt"
echo "  Fold detail: ./mmpc_sensitivity_folds/fold_ids_detail.csv"
echo "========================================================"
