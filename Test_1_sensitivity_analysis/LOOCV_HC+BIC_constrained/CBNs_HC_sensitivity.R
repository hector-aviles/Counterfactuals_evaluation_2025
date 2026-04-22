#!/usr/bin/env Rscript
# =============================================================================
# CBNs_HC_sensitivity.R
#
# Sensitivity analysis over HC+BIC hyperparameters with expert blacklist.
# All combinations MUST use the same 78 stratified fold IDs and the same
# training subsample (25% by default) for valid cross-comparison.
#
# Fold selection (Option B1):
#   - On first run: draws 78 stratified folds (13 per action) from dt_unique
#     using FOLD_SEED, saves them to ./hc_sensitivity_folds/fold_ids.txt
#   - On subsequent runs: reads fold IDs from that file — no regeneration.
#
# Training subsampling:
#   - For each fold, a stratified subsample of the remaining dt is drawn
#     (default 25%) using TRAIN_SEED, reset at the start of every combination's
#     fold loop — guaranteeing identical subsamples across all 7 combinations.
#
# Usage:
#   Rscript CBNs_HC_sensitivity.R <input_dir> \
#     --restart 0 --perturb 1 --maxp Inf --score bic [--fraction 0.25]
#
# Parameters:
#   --restart  : number of random restarts for HC      (default 0)
#   --perturb  : number of perturbations per restart   (default 1)
#   --maxp     : maximum number of parents per node    (default Inf)
#   --score    : scoring criterion: "bic" or "aic"     (default bic)
#   --fraction : fraction of remaining data to use     (default 0.25)
#
# Output structure:
#   ./<combo_label>/fold_1/cBN.dot
#   ./<combo_label>/fold_1/cBN.pl
#   ./<combo_label>/fold_1/cBN.net
#   ./<combo_label>/fold_1/test_data.csv
#   ./<combo_label>/training_numeralia.txt
#   ./<combo_label>/global_summary.csv
#   ./hc_sensitivity_folds/fold_ids.txt        (shared across all combinations)
#   ./hc_sensitivity_folds/fold_ids_detail.csv (human-readable fold metadata)
# =============================================================================

suppressMessages({
  for (pkg in c("data.table","conflicted","bnlearn","dplyr","stringr","tidyr","tidyverse","arules")) {
    if (!requireNamespace(pkg, quietly=TRUE))
      install.packages(pkg, repos="https://cloud.r-project.org")
  }
})

library(data.table); library(conflicted)
conflict_prefer("setdiff", "base")
library(bnlearn); library(dplyr); library(stringr)
library(tidyr);   library(tidyverse); library(arules)

dec_round      <- 7L
FOLD_SEED      <- 450L
N_FOLDS        <- 78L   # 13 per action x 6 actions
FOLDS_DIR      <- "./hc_sensitivity_folds"
FOLDS_FILE     <- file.path(FOLDS_DIR, "fold_ids.txt")
TRAIN_SEED     <- 450L  # reset before each combination's fold loop
DEFAULT_FRAC   <- 0.25  # default training fraction

# =============================================================================
# Argument parsing
# =============================================================================
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 1)
  stop("Usage: Rscript CBNs_HC_sensitivity.R <input_dir> [--restart N] [--perturb N] [--maxp N] [--score S] [--fraction F]")

input_dir   <- args[1]
hc_restart  <- 0L
hc_perturb  <- 1L
hc_maxp     <- Inf
hc_score    <- "bic"
hc_fraction <- DEFAULT_FRAC

named <- if (length(args) > 1) args[2:length(args)] else character(0)
i <- 1L
while (i <= length(named)) {
  fl <- named[i]
  if      (fl=="--restart"  && i<length(named)){ hc_restart  <- as.integer(named[i+1]); i<-i+2L }
  else if (fl=="--perturb"  && i<length(named)){ hc_perturb  <- as.integer(named[i+1]); i<-i+2L }
  else if (fl=="--maxp"     && i<length(named)){
    hc_maxp <- if (tolower(named[i+1]) %in% c("inf","infinity")) Inf else as.numeric(named[i+1])
    i<-i+2L
  }
  else if (fl=="--score"    && i<length(named)){ hc_score    <- named[i+1];             i<-i+2L }
  else if (fl=="--fraction" && i<length(named)){ hc_fraction <- as.numeric(named[i+1]); i<-i+2L }
  else { i<-i+1L }
}

if (!hc_score %in% c("bic","aic"))
  stop("Invalid --score '", hc_score, "'. Valid: bic, aic")
if (hc_fraction <= 0 || hc_fraction > 1)
  stop("--fraction must be in (0, 1]")

# Build combo label for output directory
maxp_str    <- if (is.infinite(hc_maxp)) "Inf" else as.character(as.integer(hc_maxp))
frac_str    <- gsub("0\\.", "p", sprintf("%.2f", hc_fraction))  # e.g. 0.25 -> p25
combo_label <- sprintf("hc_%s_restart%d_perturb%d_maxp%s_frac%s",
                       hc_score, hc_restart, hc_perturb, maxp_str, frac_str)

base_wd      <- getwd()
results_root <- file.path(base_wd, combo_label)
if (!dir.exists(results_root)) dir.create(results_root, recursive=TRUE)

message("=== HC+BIC Sensitivity Configuration ===")
message("Input dir    : ", input_dir)
message("restart      : ", hc_restart)
message("perturb      : ", hc_perturb)
message("maxp         : ", maxp_str)
message("score        : ", hc_score)
message("fraction     : ", hc_fraction)
message("Combo label  : ", combo_label)
message("Results root : ", results_root)

# =============================================================================
# Shared paths and utilities
# =============================================================================
shared_csv_path <- "./Shared_CSVs"

adjust_values <- function(freq) {
  freq[is.na(freq)] <- 0.0
  freq[freq <= 1e-4] <- 1e-3
  s <- sum(freq)
  if (s == 0) rep(1/length(freq), length(freq))
  else if (s > 1) freq/s
  else freq
}

# =============================================================================
# Load datasets
# =============================================================================
for (f in file.path(shared_csv_path, c("complete_DB_discrete.csv","crashes.csv","no_crashes.csv")))
  if (!file.exists(f)) stop("Missing: ", f)

dt            <- fread(file.path(input_dir, "complete_DB_discrete.csv"), colClasses="character")
dt            <- dt[complete.cases(dt)]
dt_crashes    <- fread(file.path(input_dir, "crashes.csv"),    colClasses="character")
dt_no_crashes <- fread(file.path(input_dir, "no_crashes.csv"), colClasses="character")
dt_crashes[,    orig_label_lc := "True"]
dt_no_crashes[, orig_label_lc := "False"]

dt_unique <- rbindlist(list(dt_crashes, dt_no_crashes))
dt_unique[, latent_collision := "True"]

state_cols <- c("curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W")
dt[,        state_id := do.call(paste, c(.SD, sep="_")), .SDcols=state_cols]
dt_unique[, state_id := do.call(paste, c(.SD, sep="_")), .SDcols=state_cols]

message("dt rows      : ", nrow(dt))
message("dt_unique rows: ", nrow(dt_unique))

# =============================================================================
# Fold selection — Option B1
#   First run: stratified draw by action, save to file.
#   All subsequent runs: read from file unchanged.
# =============================================================================
if (file.exists(FOLDS_FILE)) {
  selected_folds <- as.integer(strsplit(readLines(FOLDS_FILE), ",")[[1]])
  message("Fold source  : existing file (", FOLDS_FILE, ") — ", length(selected_folds), " folds")
} else {
  message("Fold file not found — generating ", N_FOLDS, " stratified folds with seed ", FOLD_SEED)
  if (!dir.exists(FOLDS_DIR)) dir.create(FOLDS_DIR, recursive=TRUE)

  set.seed(FOLD_SEED)
  acts       <- unique(dt_unique$action)
  n_acts     <- length(acts)
  n_per_act  <- floor(N_FOLDS / n_acts)
  remainder  <- N_FOLDS - n_per_act * n_acts

  selected_folds <- c()
  for (a in acts) {
    idx            <- which(dt_unique$action == a)
    n_draw         <- min(n_per_act, length(idx))
    selected_folds <- c(selected_folds, sample(idx, n_draw))
  }
  # Fill remainder from unselected rows, across all actions
  if (remainder > 0) {
    remaining_idx  <- setdiff(seq_len(nrow(dt_unique)), selected_folds)
    selected_folds <- c(selected_folds,
                        sample(remaining_idx, min(remainder, length(remaining_idx))))
  }
  selected_folds <- sort(unique(selected_folds))

  writeLines(paste(selected_folds, collapse=","), FOLDS_FILE)
  message("Fold IDs saved to: ", FOLDS_FILE)

  # Also save a human-readable summary alongside the IDs
  fold_meta <- dt_unique[selected_folds, .(fold_index=selected_folds,
                                            action, state_id, orig_label_lc)]
  fwrite(fold_meta, file.path(FOLDS_DIR, "fold_ids_detail.csv"))
  message("Fold detail  : ", file.path(FOLDS_DIR, "fold_ids_detail.csv"))
}

message("Selected folds: ", paste(selected_folds, collapse=", "))

# =============================================================================
# Blacklist (expert knowledge — identical across all HC combinations)
# From CBNs_LOOCV_training.R lines 268-278
# Encodes: state variables do not cause each other, action does not cause
# state variables, and latent_collision is not a cause of anything.
# =============================================================================
bl_a    <- data.frame(from=rep("action",7),
                      to=c("curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W"))
bl_cl   <- data.frame(from=rep("curr_lane",6),
                      to=c("free_E","free_NE","free_NW","free_SE","free_SW","free_W"))
bl_e    <- data.frame(from=rep("free_E",6),
                      to=c("curr_lane","free_NE","free_NW","free_SE","free_SW","free_W"))
bl_ne   <- data.frame(from=rep("free_NE",6),
                      to=c("curr_lane","free_E","free_NW","free_SE","free_SW","free_W"))
bl_nw   <- data.frame(from=rep("free_NW",6),
                      to=c("curr_lane","free_E","free_NE","free_SE","free_SW","free_W"))
bl_se   <- data.frame(from=rep("free_SE",6),
                      to=c("curr_lane","free_E","free_NE","free_NW","free_SW","free_W"))
bl_sw   <- data.frame(from=rep("free_SW",6),
                      to=c("curr_lane","free_E","free_NE","free_NW","free_SE","free_W"))
bl_w    <- data.frame(from=rep("free_W",6),
                      to=c("curr_lane","free_E","free_NE","free_NW","free_SE","free_SW"))
bl_lcol <- data.frame(from=rep("latent_collision",8),
                      to=c("action","curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W"))
blacklist <- rbind(bl_a, bl_cl, bl_e, bl_ne, bl_nw, bl_se, bl_sw, bl_w, bl_lcol)

# =============================================================================
# Helper: write .pl probabilistic logic program from a fitted BN
# (copied verbatim from CBNs_LOOCV_training.R)
# =============================================================================
write_pl <- function(bn_fit, path) {
  output_file <- file(path, "w")
  writeLines(c("%%%%%%%%%%%%%%%%%%%%%%%%%%",
               "% Exogenous variables",
               "%%%%%%%%%%%%%%%%%%%%%%%%%%\n"), con=output_file)
  u_idx <- 0L

  # Root nodes
  for (rv in bn_fit) {
    rv_name   <- rv$node
    rv_values <- dimnames(rv$prob)[[1]]
    rv_numval <- length(rv_values)
    if (!identical(rv$parents, character(0))) next

    df2       <- as.data.frame(rv$prob)
    df2[]     <- lapply(df2, as.character)
    df2$Freq  <- adjust_values(as.numeric(df2$Freq))

    if (rv_numval == 2L && identical(tolower(rv_values), c("false","true"))) {
      u_idx <- u_idx + 1L
      writeLines(paste0(format(round(df2[2,"Freq"], dec_round), nsmall=dec_round, scientific=FALSE),
                        "::u", u_idx, ".\n", rv_name, " :- u", u_idx, ".\n"), con=output_file)
    } else {
      u_name   <- paste0("u_", rv_name)
      probs    <- sapply(seq_len(rv_numval), function(j)
                    format(round(df2[j,"Freq"], dec_round), nsmall=dec_round, scientific=FALSE))
      ad_parts <- mapply(function(p,v) paste0(p,"::",u_name,"(",v,")"), probs, df2[,1])
      writeLines(paste0(paste(ad_parts, collapse="; "), "."), con=output_file)
      writeLines(paste0(rv_name, "(V) :- ", u_name, "(V).\n"), con=output_file)
    }
  }

  # Non-root nodes
  for (rv in bn_fit) {
    rv_name   <- rv$node
    rv_values <- dimnames(rv$prob)[[1]]
    rv_numval <- length(rv_values)
    if (identical(rv$parents, character(0))) next

    df2      <- as.data.frame(rv$prob)
    df2[]    <- lapply(df2, as.character)
    df2$Freq <- as.numeric(df2$Freq)

    for (row_i in seq(1L, nrow(df2), by=rv_numval)) {
      df2$Freq[row_i:(row_i+rv_numval-1L)] <- adjust_values(df2$Freq[row_i:(row_i+rv_numval-1L)])

      if (rv_numval == 2L && identical(tolower(rv_values), c("false","true"))) {
        u_idx  <- u_idx + 1L
        head   <- paste0(format(round(df2[row_i+1L,"Freq"], dec_round), nsmall=dec_round, scientific=FALSE),
                         "::u", u_idx, ".")
        body   <- paste0(rv_name, " :- u", u_idx)
        for (k in seq(2L, ncol(df2)-1L)) {
          col_vals <- unique(as.character(unlist(df2[,k])))
          body     <- paste0(body, ", ")
          if (identical(tolower(col_vals), c("false","true"))) {
            if (identical(tolower(df2[row_i,k]), "false"))
              body <- paste0(body, "\\+ ", names(df2)[k])
            else
              body <- paste0(body, names(df2)[k])
          } else {
            body <- paste0(body, names(df2)[k], "(", df2[row_i,k], ")")
          }
        }
        writeLines(paste0(head, "\n", body, ".\n"), con=output_file)
      } else {
        u_idx    <- u_idx + 1L
        u_name   <- paste0("u", u_idx)
        probs    <- sapply(row_i:(row_i+rv_numval-1L), function(r)
                      format(round(df2[r,"Freq"], dec_round), nsmall=dec_round, scientific=FALSE))
        ad_parts <- mapply(function(p,v) paste0(p,"::",u_name,"(",v,")"), probs, rv_values)
        writeLines(paste0(paste(ad_parts, collapse="; "), "."), con=output_file)
        body <- paste0(rv_name, "(V) :- ", u_name, "(V)")
        for (k in seq(2L, ncol(df2)-1L)) {
          col_vals <- unique(as.character(unlist(df2[,k])))
          body     <- paste0(body, ", ")
          if (length(col_vals) > 2L) {
            body <- paste0(body, names(df2)[k], "(", df2[row_i,k], ")")
          } else if (identical(tolower(col_vals[1]),"false") && identical(tolower(col_vals[2]),"true")) {
            if (identical(tolower(df2[row_i,k]), "false"))
              body <- paste0(body, "\\+ ", names(df2)[k])
            else
              body <- paste0(body, names(df2)[k])
          } else {
            body <- paste0(body, names(df2)[k], "(", df2[row_i,k], ")")
          }
        }
        writeLines(paste0(body, ".\n"), con=output_file)
      }
    }
  }
  close(output_file)
}

# =============================================================================
# Global summary accumulator
# =============================================================================
global_summary <- data.table(
  Fold             = integer(),
  FoldIndex        = integer(),
  Action           = character(),
  StateID          = character(),
  SamplesRemoved   = integer(),
  TrainSampleSize  = integer(),
  TrainingTime_s   = numeric(),
  SkeletonEdges    = integer(),
  HC_restart       = integer(),
  HC_perturb       = integer(),
  HC_maxp          = character(),
  HC_score         = character(),
  HC_fraction      = numeric()
)

# =============================================================================
# Main fold loop
# Reset to TRAIN_SEED before loop so training sample draws are identical
# across all 7 combinations when each is run independently.
# =============================================================================
set.seed(TRAIN_SEED)
message("\n=== Starting fold loop | seed reset to ", TRAIN_SEED, " ===")

action_list <- c("change_to_left","change_to_right","cruise","keep","swerve_left","swerve_right")

for (fi in seq_along(selected_folds)) {
  fold_i <- selected_folds[fi]
  message("\n--- Fold ", fi, "/", length(selected_folds),
          " (dt_unique row ", fold_i, ") | ",
          combo_label, " ---")

  # Output directory for this fold
  fold_dir <- file.path(results_root, paste0("fold_", fi))
  if (!dir.exists(fold_dir)) dir.create(fold_dir, recursive=TRUE)

  # ------------------------------------------------------------------
  # Test dataset: current state-action pair replicated for all 6 actions
  # ------------------------------------------------------------------
  cur_ex  <- dt_unique[fold_i,]
  dt_test <- cur_ex[rep(1:.N, each=6L)][, iaction := action_list]
  fwrite(dt_test[, lapply(.SD, as.character), .SDcols=setdiff(names(dt_test),"state_id")],
         file.path(fold_dir, "test_data.csv"), quote=TRUE)

  # ------------------------------------------------------------------
  # Training dataset: full dt minus all rows with same state
  # ------------------------------------------------------------------
  cur_state_id <- dt_unique$state_id[fold_i]
  n_removed    <- sum(dt$state_id == cur_state_id)
  train_dt     <- dt[state_id != cur_state_id]

  if (cur_state_id %in% train_dt$state_id)
    stop("Data leakage: state_id found in training set at fold ", fi)

  message("  Removed ", n_removed, " rows matching state: ", cur_state_id)
  message("  Training rows available: ", nrow(train_dt))

  # ------------------------------------------------------------------
  # Stratified subsample: at least one row per action, then fill to
  # fraction * nrow(train_dt). Seed is reset once before the fold loop
  # so this draw is identical across all 7 combinations.
  # ------------------------------------------------------------------
  sample_size  <- max(1L, round(hc_fraction * nrow(train_dt)))
  act_unique   <- unique(train_dt$action)
  n_acts_train <- length(act_unique)

  if (sample_size < n_acts_train) {
    warning("sample_size (", sample_size, ") < number of actions (", n_acts_train,
            ") at fold ", fi, " — drawing without action guarantee")
    train_sample <- train_dt[sample(.N, sample_size)]
  } else {
    # One guaranteed row per action
    min_samp  <- train_dt[, .SD[sample(.N, 1L)], by=action]
    remaining <- sample_size - nrow(min_samp)
    if (remaining > 0) {
      extra_samp   <- train_dt[sample(.N, remaining)]
      train_sample <- rbindlist(list(min_samp, extra_samp))
    } else {
      train_sample <- min_samp
    }
  }
  train_sample_size <- nrow(train_sample)
  message("  Subsample size (", round(hc_fraction*100), "%): ", train_sample_size)

  # Convert to factors for bnlearn
  df_factor <- as.data.frame(lapply(
    train_sample[, lapply(.SD, as.character), .SDcols=setdiff(names(train_sample),"state_id")],
    factor))

  # ------------------------------------------------------------------
  # Structure learning: HC+BIC with blacklist
  # ------------------------------------------------------------------
  t0 <- Sys.time()
  net <- hc(df_factor,
            whitelist  = NULL,
            blacklist  = blacklist,
            score      = hc_score,
            restart    = hc_restart,
            perturb    = hc_perturb,
            max.iter   = Inf,
            maxp       = hc_maxp,
            optimized  = TRUE,
            debug      = FALSE)
  t_struct <- as.numeric(Sys.time() - t0, units="secs")

  sk_edges <- nrow(arcs(net))
  message("  Skeleton edges: ", sk_edges, " | structure time: ", round(t_struct,2), "s")

  # Save .dot and .ps
  ob <- file.path(fold_dir, "cBN")
  write.dot(net, file=paste0(ob, ".dot"))
  try(system(paste("dot -Tps", shQuote(paste0(ob,".dot")),
                   "-o", shQuote(paste0(ob,".ps"))), intern=TRUE), silent=TRUE)

  # ------------------------------------------------------------------
  # Parameter learning
  # ------------------------------------------------------------------
  t1    <- Sys.time()
  bnf   <- bn.fit(net, data=df_factor, method="mle", replace.unidentifiable=TRUE)
  t_tot <- t_struct + as.numeric(Sys.time() - t1, units="secs")

  # Save .net and .pl
  try(write.net(paste0(ob,".net"), bnf), silent=TRUE)
  write_pl(bnf, paste0(ob, ".pl"))

  # ------------------------------------------------------------------
  # Per-fold log entry
  # ------------------------------------------------------------------
  log_line <- sprintf(
    "Fold %d (row %d) | action=%s | state=%s | removed=%d | train_n=%d | frac=%.2f | edges=%d | time=%.2fs | restart=%d | perturb=%d | maxp=%s | score=%s",
    fi, fold_i,
    dt_unique$action[fold_i], cur_state_id,
    n_removed, train_sample_size, hc_fraction, sk_edges, t_tot,
    hc_restart, hc_perturb, maxp_str, hc_score)
  message("  ", log_line)
  write(log_line,
        file   = file.path(results_root, "training_numeralia.txt"),
        append = TRUE)

  # ------------------------------------------------------------------
  # Accumulate global summary
  # ------------------------------------------------------------------
  global_summary <- rbindlist(list(global_summary, data.table(
    Fold            = fi,
    FoldIndex       = fold_i,
    Action          = dt_unique$action[fold_i],
    StateID         = cur_state_id,
    SamplesRemoved  = as.integer(n_removed),
    TrainSampleSize = as.integer(train_sample_size),
    TrainingTime_s  = t_tot,
    SkeletonEdges   = as.integer(sk_edges),
    HC_restart      = as.integer(hc_restart),
    HC_perturb      = as.integer(hc_perturb),
    HC_maxp         = maxp_str,
    HC_score        = hc_score,
    HC_fraction     = hc_fraction
  )))

  rm(train_dt, train_sample, df_factor, bnf, net); gc()
} # end fold loop

# =============================================================================
# Save global summary CSV and print final statistics
# =============================================================================
csv_path <- file.path(results_root, "global_summary.csv")
fwrite(global_summary, csv_path)
message("\nGlobal summary saved to: ", csv_path)

message("\n=== FINAL SUMMARY: ", combo_label, " ===")
message(sprintf("Folds completed  : %d", nrow(global_summary)))
message(sprintf("Avg training time: %.2f ± %.2f s",
                mean(global_summary$TrainingTime_s),
                sd(global_summary$TrainingTime_s)))
message(sprintf("Avg skeleton edges: %.1f ± %.1f",
                mean(global_summary$SkeletonEdges),
                sd(global_summary$SkeletonEdges)))
message(sprintf("Avg samples removed: %.1f ± %.1f",
                mean(global_summary$SamplesRemoved),
                sd(global_summary$SamplesRemoved)))

message("\nAction distribution across selected folds:")
print(global_summary[, .N, by=Action][order(Action)])

# Append summary block to numeralia file
summ_str <- sprintf(
  "\n=== SUMMARY: %s ===\nFolds: %d\nAvg time: %.2f +/- %.2f s\nAvg edges: %.1f +/- %.1f\nAvg removed: %.1f +/- %.1f\n",
  combo_label,
  nrow(global_summary),
  mean(global_summary$TrainingTime_s), sd(global_summary$TrainingTime_s),
  mean(global_summary$SkeletonEdges),  sd(global_summary$SkeletonEdges),
  mean(global_summary$SamplesRemoved), sd(global_summary$SamplesRemoved))
write(summ_str, file=file.path(results_root,"training_numeralia.txt"), append=TRUE)

message("\nScript finished.")
