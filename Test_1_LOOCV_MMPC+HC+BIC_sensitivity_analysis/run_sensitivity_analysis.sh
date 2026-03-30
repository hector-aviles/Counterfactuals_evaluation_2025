#!/bin/bash
# =============================================================================
# run_sensitivity_analysis.sh
#
# Runs all 7 MMPC sensitivity combinations on the SAME 30 stratified folds,
# guaranteeing directly comparable results.
#
# Workflow:
#   1. Generate 30 stratified fold IDs once (seed=42, fixed)
#   2. Run all 7 combinations passing the SAME --fold_ids string
#   3. Merge all global_summary.csv files into one consolidated file
#   4. Print a comparison table broken down by training percentage
#
# Output structure:
#   sensitivity_results/
#     fold_ids_used.txt                    <- the 30 fold IDs (shared)
#     A_mi_alpha0.01_maxsx3/
#       rep_1/01/cBNs/  rep_1/50/cBNs/  rep_1/100/cBNs/
#       global_summary.csv
#       fold_ids_used.txt                  <- copy for traceability
#     B_mi_alpha0.05_maxsx3_BASELINE/
#       ...
#     G_mi_sh_alpha0.05_maxsx3/
#       ...
#     all_combinations_summary.csv         <- merged from all 7
#     comparison_table.txt                 <- human-readable comparison
#
# Usage:
#   chmod +x run_sensitivity_analysis.sh
#   ./run_sensitivity_analysis.sh
#
# Background with full log:
#   nohup ./run_sensitivity_analysis.sh > sensitivity_master.log 2>&1 &
# =============================================================================

SCRIPT="CBNs_LOOCV_training_MMPC_sensitivity.R"
INPUT_DIR="./Shared_CSVs"
REP="1"
PCT="01,50,100"
N_FOLDS=30
SEED=42
RESULTS_DIR="sensitivity_results"

mkdir -p "$RESULTS_DIR"

echo "=============================================================="
echo " MMPC Sensitivity Analysis"
echo " Script      : $SCRIPT"
echo " Input       : $INPUT_DIR"
echo " Rep         : $REP"
echo " Percentages : $PCT"
echo " Folds       : $N_FOLDS (stratified, seed=$SEED)"
echo " Started     : $(date)"
echo "=============================================================="

# =============================================================================
# STEP 1 — Generate fixed stratified fold IDs once
# All combinations will receive exactly these IDs via --fold_ids
# =============================================================================
echo ""
echo "[STEP 1] Generating $N_FOLDS stratified fold IDs (seed=$SEED)..."

FOLD_IDS=$(Rscript - "$INPUT_DIR" "$N_FOLDS" "$SEED" <<'REOF'
args      <- commandArgs(trailingOnly=TRUE)
input_dir <- args[1]
n_folds   <- as.integer(args[2])
seed      <- as.integer(args[3])

suppressMessages(library(data.table))

dt_crashes    <- fread(file.path(input_dir,"crashes.csv"),    colClasses="character")
dt_no_crashes <- fread(file.path(input_dir,"no_crashes.csv"), colClasses="character")
dt_crashes[,    orig_label_lc:="True"]
dt_no_crashes[, orig_label_lc:="False"]
dt_unique <- rbindlist(list(dt_crashes,dt_no_crashes))
dt_unique[, latent_collision:="True"]

set.seed(seed)
acts    <- unique(dt_unique$action)
n_per   <- max(1L, floor(n_folds/length(acts)))
folds   <- c()
for (a in acts) {
  idx   <- which(dt_unique$action==a)
  folds <- c(folds, sample(idx, min(n_per,length(idx))))
}
remain <- setdiff(seq_len(nrow(dt_unique)), folds)
short  <- n_folds - length(folds)
if (short>0 && length(remain)>0)
  folds <- c(folds, sample(remain, min(short,length(remain))))
folds <- sort(unique(folds))

# Print action distribution for the selected folds
cat(sprintf("Selected %d folds. Action distribution:\n", length(folds)), file=stderr())
dist <- table(dt_unique$action[folds])
for (a in names(dist)) cat(sprintf("  %-20s : %d\n", a, dist[a]), file=stderr())

cat(paste(folds, collapse=","))
REOF
)

if [ -z "$FOLD_IDS" ]; then
  echo "ERROR: Failed to generate fold IDs. Aborting."
  exit 1
fi

echo "Fold IDs: $FOLD_IDS"
echo "$FOLD_IDS" > "${RESULTS_DIR}/fold_ids_used.txt"
echo "Saved to: ${RESULTS_DIR}/fold_ids_used.txt"

# =============================================================================
# STEP 2 — Define the 7 combinations
# Format: "LABEL|TEST|ALPHA|MAX_SX|PURPOSE"
# =============================================================================
COMBOS=(
  "A_mi_alpha0.01_maxsx3|mi|0.01|3|Stricter alpha (fewer edges admitted)"
  "B_mi_alpha0.05_maxsx3_BASELINE|mi|0.05|3|Baseline — matches paper"
  "C_mi_alpha0.10_maxsx3|mi|0.10|3|More lenient alpha"
  "D_mi_alpha0.20_maxsx3|mi|0.20|3|Much more lenient alpha"
  "E_mi_alpha0.05_maxsx2|mi|0.05|2|Smaller conditioning set"
  "F_mi_alpha0.05_maxsx4|mi|0.05|4|Larger conditioning set"
  "G_mi_sh_alpha0.05_maxsx3|mi-sh|0.05|3|Shrinkage MI estimator (bias-corrected)"
)

echo ""
echo "[STEP 2] Running ${#COMBOS[@]} combinations on the same $N_FOLDS folds..."

# =============================================================================
# STEP 3 — Run each combination
# =============================================================================
declare -A COMBO_STATUS

for entry in "${COMBOS[@]}"; do
  IFS="|" read -r LABEL TEST ALPHA MAXSX PURPOSE <<< "$entry"

  echo ""
  echo "--------------------------------------------------------------"
  echo "Combination : $LABEL"
  echo "Purpose     : $PURPOSE"
  echo "Parameters  : test=$TEST  alpha=$ALPHA  max.sx=$MAXSX"
  echo "Folds       : same $N_FOLDS IDs as all other combos"
  echo "Started     : $(date)"
  echo "--------------------------------------------------------------"

  LOG_FILE="${RESULTS_DIR}/${LABEL}.log"

  COMBO_LABEL="$LABEL" Rscript "$SCRIPT" \
    "$INPUT_DIR" "$REP" "$PCT" \
    --alpha  "$ALPHA" \
    --max.sx "$MAXSX" \
    --test   "$TEST"  \
    --fold_ids "$FOLD_IDS" \
    > "$LOG_FILE" 2>&1

  EXIT_CODE=$?
  COMBO_STATUS[$LABEL]=$EXIT_CODE

  if [ $EXIT_CODE -eq 0 ]; then
    echo "  COMPLETED : $(date)"
    # Show per-percentage summary lines from the log
    grep "^\[Pct\|Pct=" "$LOG_FILE" | sed 's/^/  /' 2>/dev/null || true
    grep "action_in_LC=" "$LOG_FILE" | grep "Pct=" | tail -3 | sed 's/^/  /' 2>/dev/null || true
  else
    echo "  FAILED (exit=$EXIT_CODE) : $(date)"
    echo "  Last 5 lines of log:"
    tail -5 "$LOG_FILE" | sed 's/^/    /'
  fi
done

# =============================================================================
# STEP 4 — Merge all global_summary.csv into one file
# =============================================================================
echo ""
echo "[STEP 4] Merging CSVs..."

MERGED="${RESULTS_DIR}/all_combinations_summary.csv"
FIRST_CSV=$(ls ${RESULTS_DIR}/*/global_summary.csv 2>/dev/null | head -1)

if [ -n "$FIRST_CSV" ]; then
  head -1 "$FIRST_CSV" > "$MERGED"
  for csv in ${RESULTS_DIR}/*/global_summary.csv; do
    tail -n +2 "$csv" >> "$MERGED"
  done
  TOTAL_ROWS=$(wc -l < "$MERGED")
  echo "Merged: $MERGED ($TOTAL_ROWS lines including header)"
else
  echo "WARNING: No global_summary.csv files found."
fi

# =============================================================================
# STEP 5 — Comparison table broken down by training percentage
# =============================================================================
echo ""
echo "[STEP 5] Generating comparison table..."

CMP_FILE="${RESULTS_DIR}/comparison_table.txt"

Rscript - "$MERGED" "$N_FOLDS" <<'REOF' | tee "$CMP_FILE"
args    <- commandArgs(trailingOnly=TRUE)
csv     <- args[1]; n_folds <- as.integer(args[2])
suppressMessages(library(data.table))

if (!file.exists(csv)) { cat("Merged CSV not found.\n"); quit(status=1) }
d <- fread(csv)
if (nrow(d)==0) { cat("No data in merged CSV.\n"); quit(status=0) }

cat("\n")
cat("================================================================\n")
cat(" MMPC SENSITIVITY ANALYSIS — COMPARISON TABLE\n")
cat(sprintf(" Folds per combination per percentage: ~%d\n", n_folds))
cat(sprintf(" Training percentages: %s\n", paste(sort(unique(d$Percentage)),collapse=", ")))
cat("================================================================\n\n")

pcts <- sort(unique(d$Percentage))

for (pct in pcts) {
  sub <- d[Percentage==pct]

  cat(sprintf("--- Training percentage: %s%% ---\n", as.integer(pct)))
  cat(sprintf("%-42s %-8s %-6s %-6s  %14s  %14s  %10s  %10s\n",
              "Combination","Test","Alpha","MaxSX",
              "action_in_LC","skel_connected","avg_comps","avg_time_s"))
  cat(paste(rep("-",110),collapse=""),"\n")

  combos_in_data <- unique(sub[, paste(MMPC_test, MMPC_alpha, MMPC_max_sx)])
  summ <- sub[, .(
    MMPC_test    = unique(MMPC_test),
    MMPC_alpha   = unique(MMPC_alpha),
    MMPC_max_sx  = unique(MMPC_max_sx),
    n            = .N,
    act_n        = sum(Action_in_LC_neighbors),
    act_pct      = round(100*mean(Action_in_LC_neighbors),1),
    con_n        = sum(Skeleton_connected),
    con_pct      = round(100*mean(Skeleton_connected),1),
    avg_comp     = round(mean(Skeleton_components),2),
    avg_time     = round(mean(TrainingTime_s),2)
  ), by=.(combo_label=paste0(MMPC_test,"_a",MMPC_alpha,"_sx",MMPC_max_sx))]

  # Try to pull proper labels from directory names
  labels <- unique(d[Percentage==pct, .(MMPC_test, MMPC_alpha, MMPC_max_sx)])

  for (j in seq_len(nrow(summ))) {
    r  <- summ[j]
    lbl <- sprintf("%s|a=%.2f|sx=%d", r$MMPC_test, r$MMPC_alpha, r$MMPC_max_sx)
    act_str <- sprintf("%d/%d (%.1f%%)", r$act_n, r$n, r$act_pct)
    con_str <- sprintf("%d/%d (%.1f%%)", r$con_n, r$n, r$con_pct)
    cat(sprintf("%-42s %-8s %-6.2f %-6d  %14s  %14s  %10.2f  %10.2f\n",
                lbl, r$MMPC_test, r$MMPC_alpha, r$MMPC_max_sx,
                act_str, con_str, r$avg_comp, r$avg_time))
  }
  cat("\n")
}

# ---- Top LC neighbor patterns per combination and percentage ----
cat("================================================================\n")
cat(" LC NEIGHBOR PATTERNS (top 4 per combination × percentage)\n")
cat("================================================================\n\n")

for (pct in pcts) {
  cat(sprintf("--- Training percentage: %s%% ---\n", as.integer(pct)))
  sub <- d[Percentage==pct]
  for (combo in unique(sub[,paste(MMPC_test,MMPC_alpha,MMPC_max_sx)])) {
    parts <- strsplit(combo," ")[[1]]
    tst<-parts[1]; alp<-as.numeric(parts[2]); sx<-as.integer(parts[3])
    csub <- sub[MMPC_test==tst & MMPC_alpha==alp & MMPC_max_sx==sx]
    lf   <- sort(table(csub$LC_neighbors), decreasing=TRUE)
    cat(sprintf("  %s | alpha=%.2f | max.sx=%d  (n=%d folds)\n",tst,alp,sx,nrow(csub)))
    for (k in seq_len(min(4L,length(lf))))
      cat(sprintf("    %-40s : %d (%.1f%%)\n", names(lf)[k], lf[k], 100*lf[k]/nrow(csub)))
  }
  cat("\n")
}

# ---- Timing summary ----
cat("================================================================\n")
cat(" TRAINING TIME SUMMARY (seconds)\n")
cat("================================================================\n")
cat(sprintf("%-42s  %10s  %10s  %10s\n","Combination","mean","sd","min","max"))
# fix header
cat(sprintf("%-42s  %10s  %10s  %10s  %10s\n","Combination","mean_s","sd_s","min_s","max_s"))
cat(paste(rep("-",85),collapse=""),"\n")
ts <- d[, .(mean_s=round(mean(TrainingTime_s),2), sd_s=round(sd(TrainingTime_s),2),
            min_s=round(min(TrainingTime_s),2), max_s=round(max(TrainingTime_s),2)),
        by=.(MMPC_test,MMPC_alpha,MMPC_max_sx)]
for (j in seq_len(nrow(ts))) {
  r   <- ts[j]
  lbl <- sprintf("%s|a=%.2f|sx=%d",r$MMPC_test,r$MMPC_alpha,r$MMPC_max_sx)
  cat(sprintf("%-42s  %10.2f  %10.2f  %10.2f  %10.2f\n",
              lbl, r$mean_s, r$sd_s, r$min_s, r$max_s))
}
cat("\n")
REOF

echo ""
echo "Comparison table saved: $CMP_FILE"

# =============================================================================
# STEP 6 — Final status report
# =============================================================================
echo ""
echo "=============================================================="
echo " FINAL STATUS"
echo "=============================================================="
echo "Fold IDs used (all combos): ${RESULTS_DIR}/fold_ids_used.txt"
echo "Merged CSV               : ${MERGED}"
echo "Comparison table         : ${CMP_FILE}"
echo ""
echo "Combination results:"
for entry in "${COMBOS[@]}"; do
  IFS="|" read -r LABEL TEST ALPHA MAXSX PURPOSE <<< "$entry"
  STATUS=${COMBO_STATUS[$LABEL]:-"not run"}
  if [ "$STATUS" = "0" ]; then
    STATUS_STR="OK"
  else
    STATUS_STR="FAILED (exit=$STATUS)"
  fi
  printf "  %-45s %s\n" "$LABEL" "$STATUS_STR"
done
echo ""
echo "All done: $(date)"
