#!/usr/bin/env bash
# =============================================================================
# run_hc_sensitivity.sh
#
# Runs all 7 HC+BIC parameter combinations sequentially.
# Fold IDs are generated on the first call and reused by all subsequent calls.
#
# Usage:
#   chmod +x run_hc_sensitivity.sh
#   ./run_hc_sensitivity.sh <input_dir> 2>&1 | tee hc_sensitivity_master.log
# =============================================================================

set -euo pipefail

INPUT_DIR="${1:?Usage: ./run_hc_sensitivity.sh <input_dir>}"
SCRIPT="CBNs_HC_sensitivity.R"

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

# | Config | restart | perturb | maxp | score |
# |--------|---------|---------|------|-------|
# | 1      | 0       | 1       | Inf  | bic   |
# | 2      | 5       | 1       | Inf  | bic   |
# | 3      | 10      | 1       | Inf  | bic   |
# | 4      | 0       | 1       | 3    | bic   |
# | 5      | 0       | 1       | 5    | bic   |
# | 6      | 0       | 2       | Inf  | bic   |
# | 7      | 0       | 1       | Inf  | aic   |

run_combo "config_1_restart0_perturb1_maxpInf_bic"  --restart 0  --perturb 1 --maxp Inf --score bic --fraction 0.25
run_combo "config_2_restart5_perturb1_maxpInf_bic"  --restart 5  --perturb 1 --maxp Inf --score bic --fraction 0.25
run_combo "config_3_restart10_perturb1_maxpInf_bic" --restart 10 --perturb 1 --maxp Inf --score bic --fraction 0.25
run_combo "config_4_restart0_perturb1_maxp3_bic"    --restart 0  --perturb 1 --maxp 3   --score bic --fraction 0.25
run_combo "config_5_restart0_perturb1_maxp5_bic"    --restart 0  --perturb 1 --maxp 5   --score bic --fraction 0.25
run_combo "config_6_restart0_perturb2_maxpInf_bic"  --restart 0  --perturb 2 --maxp Inf --score bic --fraction 0.25
run_combo "config_7_restart0_perturb1_maxpInf_aic"  --restart 0  --perturb 1 --maxp Inf --score aic --fraction 0.25

echo ""
echo "========================================================"
echo "  All combinations completed: $(date)"
echo "  Fold IDs used: ./hc_sensitivity_folds/fold_ids.txt"
echo "  Fold detail  : ./hc_sensitivity_folds/fold_ids_detail.csv"
echo "========================================================"
