#!/usr/bin/env Rscript
# =============================================================================
# CBNs_MMPC_sensitivity.R
#
# Sensitivity analysis over MMPC+HC+BIC hyperparameters.
# All combinations MUST use the same 78 stratified fold IDs and the same
# training subsample (25% by default) for valid cross-comparison.
#
# Fold selection (Option B1):
#   - On first run: draws 78 stratified folds (13 per action) from dt_unique
#     using FOLD_SEED, saves them to ./mmpc_sensitivity_folds/fold_ids.txt
#   - On subsequent runs: reads fold IDs from that file — no regeneration.
#
# Training subsampling:
#   - For each fold, a stratified subsample of the remaining dt is drawn
#     (default 25%) using TRAIN_SEED, reset at the start of every combination's
#     fold loop — guaranteeing identical subsamples across all 7 combinations.
#
# Usage:
#   Rscript CBNs_MMPC_sensitivity.R <input_dir> \
#     --alpha 0.05 --max.sx 3 --test mi [--fraction 0.25]
#
# Parameters:
#   --alpha    : CI test significance threshold   (default 0.05)
#   --max.sx   : maximum conditioning set size    (default 3)
#   --test     : CI test: mi, x2, mi-sh, etc.    (default mi)
#   --fraction : fraction of remaining data to use (default 0.25)
#
# Output structure:
#   ./<combo_label>/fold_1/cBN.dot
#   ./<combo_label>/fold_1/cBN.pl
#   ./<combo_label>/fold_1/cBN.net
#   ./<combo_label>/fold_1/test_data.csv
#   ./<combo_label>/training_numeralia.txt
#   ./<combo_label>/global_summary.csv
#   ./mmpc_sensitivity_folds/fold_ids.txt        (shared across all combinations)
#   ./mmpc_sensitivity_folds/fold_ids_detail.csv (human-readable fold metadata)
#
# Parameter combinations:
#   Config 1 (baseline): alpha=0.05, max.sx=3, test=mi
#   Config 2:            alpha=0.01, max.sx=3, test=mi
#   Config 3:            alpha=0.10, max.sx=3, test=mi
#   Config 4:            alpha=0.05, max.sx=2, test=mi
#   Config 5:            alpha=0.05, max.sx=4, test=mi
#   Config 6:            alpha=0.01, max.sx=2, test=mi
#   Config 7:            alpha=0.05, max.sx=3, test=x2
# =============================================================================

suppressMessages({
  for (pkg in c("data.table","conflicted","bnlearn","dplyr","stringr",
                "tidyr","tidyverse","arules")) {
    if (!requireNamespace(pkg, quietly=TRUE))
      install.packages(pkg, repos="https://cloud.r-project.org")
  }
})

library(data.table); library(conflicted)
conflict_prefer("setdiff", "base")
library(bnlearn); library(dplyr); library(stringr)
library(tidyr);   library(tidyverse); library(arules)

dec_round    <- 7L
FOLD_SEED    <- 450L
N_FOLDS      <- 78L   # 13 per action x 6 actions
FOLDS_DIR    <- "./mmpc_sensitivity_folds"
FOLDS_FILE   <- file.path(FOLDS_DIR, "fold_ids.txt")
TRAIN_SEED   <- 450L  # reset before each combination's fold loop
DEFAULT_FRAC <- 0.25  # default training fraction

# =============================================================================
# Argument parsing
# =============================================================================
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 1)
  stop("Usage: Rscript CBNs_MMPC_sensitivity.R <input_dir> ",
       "[--alpha V] [--max.sx V] [--test T] [--fraction F]")

input_dir    <- args[1]
mmpc_alpha   <- 0.05
mmpc_max_sx  <- 3L
mmpc_test    <- "mi"
hc_fraction  <- DEFAULT_FRAC

named <- if (length(args) > 1) args[2:length(args)] else character(0)
i <- 1L
while (i <= length(named)) {
  fl <- named[i]
  if      (fl=="--alpha"    && i<length(named)){ mmpc_alpha   <- as.numeric(named[i+1]);  i<-i+2L }
  else if (fl=="--max.sx"   && i<length(named)){ mmpc_max_sx  <- as.integer(named[i+1]); i<-i+2L }
  else if (fl=="--test"     && i<length(named)){ mmpc_test    <- named[i+1];              i<-i+2L }
  else if (fl=="--fraction" && i<length(named)){ hc_fraction  <- as.numeric(named[i+1]); i<-i+2L }
  else { i<-i+1L }
}

valid_tests <- c("mi","mi-sh","x2","sp-mi","smc-mi")
if (!mmpc_test %in% valid_tests)
  stop("Invalid --test '", mmpc_test, "'. Valid: ", paste(valid_tests, collapse=", "))
if (hc_fraction <= 0 || hc_fraction > 1)
  stop("--fraction must be in (0, 1]")

# Build combo label for output directory
alpha_str   <- gsub("\\.", "p", sprintf("%.2f", mmpc_alpha))  # 0.05 -> 0p05
frac_str    <- gsub("0\\.", "p", sprintf("%.2f", hc_fraction)) # 0.25 -> p25
test_str    <- gsub("-", "_", mmpc_test)                        # mi-sh -> mi_sh
combo_label <- sprintf("mmpc_%s_alpha%s_maxsx%d_frac%s",
                       test_str, alpha_str, mmpc_max_sx, frac_str)

base_wd      <- getwd()
results_root <- file.path(base_wd, combo_label)
if (!dir.exists(results_root)) dir.create(results_root, recursive=TRUE)

message("=== MMPC+HC+BIC Sensitivity Configuration ===")
message("Input dir    : ", input_dir)
message("alpha        : ", mmpc_alpha)
message("max.sx       : ", mmpc_max_sx)
message("test         : ", mmpc_test)
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
for (f in file.path(shared_csv_path,
                    c("complete_DB_discrete.csv","crashes.csv","no_crashes.csv")))
  if (!file.exists(f)) stop("Missing: ", f)

dt            <- fread(file.path(input_dir, "complete_DB_discrete.csv"),
                       colClasses="character")
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

message("dt rows       : ", nrow(dt))
message("dt_unique rows: ", nrow(dt_unique))

# =============================================================================
# Fold selection — Option B1
#   First run: stratified draw by action, save to file.
#   All subsequent runs: read from file unchanged.
# =============================================================================
if (file.exists(FOLDS_FILE)) {
  selected_folds <- as.integer(strsplit(readLines(FOLDS_FILE), ",")[[1]])
  message("Fold source  : existing file (", FOLDS_FILE, ") — ",
          length(selected_folds), " folds")
} else {
  message("Fold file not found — generating ", N_FOLDS,
          " stratified folds with seed ", FOLD_SEED)
  if (!dir.exists(FOLDS_DIR)) dir.create(FOLDS_DIR, recursive=TRUE)

  set.seed(FOLD_SEED)
  acts      <- unique(dt_unique$action)
  n_acts    <- length(acts)
  n_per_act <- floor(N_FOLDS / n_acts)
  remainder <- N_FOLDS - n_per_act * n_acts

  selected_folds <- c()
  for (a in acts) {
    idx            <- which(dt_unique$action == a)
    n_draw         <- min(n_per_act, length(idx))
    selected_folds <- c(selected_folds, sample(idx, n_draw))
  }
  if (remainder > 0) {
    remaining_idx  <- setdiff(seq_len(nrow(dt_unique)), selected_folds)
    selected_folds <- c(selected_folds,
                        sample(remaining_idx,
                               min(remainder, length(remaining_idx))))
  }
  selected_folds <- sort(unique(selected_folds))

  writeLines(paste(selected_folds, collapse=","), FOLDS_FILE)
  message("Fold IDs saved to: ", FOLDS_FILE)

  fold_meta <- dt_unique[selected_folds,
                          .(fold_index=selected_folds,
                            action, state_id, orig_label_lc)]
  fwrite(fold_meta, file.path(FOLDS_DIR, "fold_ids_detail.csv"))
  message("Fold detail  : ", file.path(FOLDS_DIR, "fold_ids_detail.csv"))
}

message("Selected folds: ", paste(selected_folds, collapse=", "))

# =============================================================================
# Helper: write .pl probabilistic logic program from a fitted BN
# (identical to CBNs_LOOCV_training.R)
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

    df2      <- as.data.frame(rv$prob)
    df2[]    <- lapply(df2, as.character)
    df2$Freq <- adjust_values(as.numeric(df2$Freq))

    if (rv_numval == 2L && identical(tolower(rv_values), c("false","true"))) {
      u_idx <- u_idx + 1L
      writeLines(paste0(format(round(df2[2,"Freq"], dec_round),
                               nsmall=dec_round, scientific=FALSE),
                        "::u", u_idx, ".\n", rv_name, " :- u", u_idx, ".\n"),
                 con=output_file)
    } else {
      u_name   <- paste0("u_", rv_name)
      probs    <- sapply(seq_len(rv_numval), function(j)
        format(round(df2[j,"Freq"], dec_round), nsmall=dec_round, scientific=FALSE))
      ad_parts <- mapply(function(p,v) paste0(p,"::",u_name,"(",v,")"),
                         probs, df2[,1])
      writeLines(paste0(paste(ad_parts, collapse="; "), "."), con=output_file)
      writeLines(paste0(rv_name, "(V) :- ", u_name, "(V).\n"),   con=output_file)
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
      df2$Freq[row_i:(row_i+rv_numval-1L)] <-
        adjust_values(df2$Freq[row_i:(row_i+rv_numval-1L)])

      if (rv_numval == 2L && identical(tolower(rv_values), c("false","true"))) {
        u_idx <- u_idx + 1L
        head  <- paste0(format(round(df2[row_i+1L,"Freq"], dec_round),
                               nsmall=dec_round, scientific=FALSE),
                        "::u", u_idx, ".")
        body  <- paste0(rv_name, " :- u", u_idx)
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
        ad_parts <- mapply(function(p,v) paste0(p,"::",u_name,"(",v,")"),
                           probs, rv_values)
        writeLines(paste0(paste(ad_parts, collapse="; "), "."), con=output_file)
        body <- paste0(rv_name, "(V) :- ", u_name, "(V)")
        for (k in seq(2L, ncol(df2)-1L)) {
          col_vals <- unique(as.character(unlist(df2[,k])))
          body     <- paste0(body, ", ")
          if (length(col_vals) > 2L) {
            body <- paste0(body, names(df2)[k], "(", df2[row_i,k], ")")
          } else if (identical(tolower(col_vals[1]),"false") &&
                     identical(tolower(col_vals[2]),"true")) {
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
  Fold                  = integer(),
  FoldIndex             = integer(),
  Action                = character(),
  StateID               = character(),
  SamplesRemoved        = integer(),
  TrainSampleSize       = integer(),
  TrainingTime_s        = numeric(),
  MMPC_alpha            = numeric(),
  MMPC_max_sx           = integer(),
  MMPC_test             = character(),
  MMPC_skeleton_edges   = integer(),
  MMPC_sk_connected     = logical(),
  MMPC_sk_components    = integer(),
  MMPC_LC_neighbors     = character(),
  MMPC_action_in_LC_nbr = logical(),
  HC_final_arcs         = integer(),
  HC_fraction           = numeric()
)

# =============================================================================
# Main fold loop
# Reset to TRAIN_SEED before loop so training sample draws are identical
# across all 7 combinations when each is run independently.
# =============================================================================
set.seed(TRAIN_SEED)
message("\n=== Starting fold loop | seed reset to ", TRAIN_SEED, " ===")

action_list <- c("change_to_left","change_to_right","cruise",
                 "keep","swerve_left","swerve_right")

for (fi in seq_along(selected_folds)) {
  fold_i <- selected_folds[fi]
  message("\n--- Fold ", fi, "/", length(selected_folds),
          " (dt_unique row ", fold_i, ") | ", combo_label, " ---")

  # Output directory for this fold
  fold_dir <- file.path(results_root, paste0("fold_", fi))
  if (!dir.exists(fold_dir)) dir.create(fold_dir, recursive=TRUE)

  # ------------------------------------------------------------------
  # Test dataset: current state-action pair replicated for all 6 actions
  # ------------------------------------------------------------------
  cur_ex  <- dt_unique[fold_i,]
  dt_test <- cur_ex[rep(1:.N, each=6L)][, iaction := action_list]
  fwrite(dt_test[, lapply(.SD, as.character),
                 .SDcols=setdiff(names(dt_test),"state_id")],
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
  # Stratified subsample (identical across all combinations via TRAIN_SEED)
  # ------------------------------------------------------------------
  sample_size  <- max(1L, round(hc_fraction * nrow(train_dt)))
  act_unique   <- unique(train_dt$action)
  n_acts_train <- length(act_unique)

  if (sample_size < n_acts_train) {
    warning("sample_size (", sample_size, ") < number of actions (",
            n_acts_train, ") at fold ", fi, " — drawing without action guarantee")
    train_sample <- train_dt[sample(.N, sample_size)]
  } else {
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
    train_sample[, lapply(.SD, as.character),
                 .SDcols=setdiff(names(train_sample),"state_id")],
    factor))

  # ------------------------------------------------------------------
  # MMPC: learn skeleton
  # ------------------------------------------------------------------
  t0 <- Sys.time()

  net_mmpc <- mmpc(df_factor,
                   test   = mmpc_test,
                   alpha  = mmpc_alpha,
                   max.sx = mmpc_max_sx)

  # --- LC neighbor diagnostics (mirrors CBNs_LOOCV_training.R) ---
  lc_nbrs   <- tryCatch(bnlearn::nbr(net_mmpc, "latent_collision"),
                        error = function(e) character(0))
  act_in_lc <- "action" %in% lc_nbrs
  lc_str    <- if (length(lc_nbrs) == 0L) "NONE"
               else paste(sort(lc_nbrs), collapse=", ")

  message("  MMPC neighbors of latent_collision: ", lc_str)
  message("  action in LC neighbors: ", act_in_lc)

  # --- Skeleton connectivity via BFS (mirrors CBNs_LOOCV_training.R) ---
  adj     <- amat(net_mmpc)
  nodes   <- rownames(adj)
  n_nodes <- length(nodes)
  visited <- rep(FALSE, n_nodes)
  comps   <- list()

  for (start in seq_len(n_nodes)) {
    if (!visited[start]) {
      queue <- start; comp <- integer(0)
      while (length(queue) > 0) {
        u <- queue[1]; queue <- queue[-1]
        if (!visited[u]) {
          visited[u] <- TRUE; comp <- c(comp, u)
          nb <- which(adj[u,] == 1)
          queue <- unique(c(queue, nb[!visited[nb]]))
        }
      }
      comps <- c(comps, list(nodes[comp]))
    }
  }

  sk_edges     <- nrow(arcs(net_mmpc))
  sk_connected <- (length(comps) == 1L)
  sk_comps     <- length(comps)

  message("  MMPC skeleton: ", sk_edges, " edges | connected=", sk_connected,
          " | components=", sk_comps)

  # --- Save MMPC skeleton summary ---
  sk_summary_file <- file.path(fold_dir, "mmpc_skeleton_summary.txt")
  cat(c(paste0("Fold ", fi, " (dt_unique row ", fold_i, ")"),
        paste0("MMPC alpha=", mmpc_alpha, " max.sx=", mmpc_max_sx,
               " test=", mmpc_test),
        paste0("LC neighbors: ", lc_str),
        paste0("action in LC neighbors: ", act_in_lc),
        paste0("Skeleton edges: ", sk_edges),
        paste0("Connected: ", sk_connected),
        paste0("Components: ", sk_comps), ""),
      file=sk_summary_file, sep="\n")

  # ------------------------------------------------------------------
  # Build blacklist from MMPC skeleton (zero entries = forbidden arcs)
  # ------------------------------------------------------------------
  missing_arcs <- which(adj == 0 & row(adj) != col(adj), arr.ind=TRUE)
  blacklist_df <- unique(data.frame(
    from = rownames(adj)[missing_arcs[,"row"]],
    to   = colnames(adj)[missing_arcs[,"col"]],
    stringsAsFactors = FALSE
  ))
  message("  Blacklisted arcs from MMPC skeleton: ", nrow(blacklist_df))

  # ------------------------------------------------------------------
  # HC+BIC with MMPC-derived blacklist
  # ------------------------------------------------------------------
  net <- hc(df_factor,
            whitelist  = NULL,
            blacklist  = blacklist_df,
            score      = "bic",
            restart    = 0L,
            perturb    = 1L,
            max.iter   = Inf,
            maxp       = Inf,
            optimized  = TRUE,
            debug      = FALSE)

  t_struct   <- as.numeric(Sys.time() - t0, units="secs")
  final_arcs <- nrow(arcs(net))
  message("  HC final arcs: ", final_arcs, " | structure time: ",
          round(t_struct,2), "s")

  # Save .dot and .ps
  ob <- file.path(fold_dir, "cBN")
  write.dot(net, file=paste0(ob, ".dot"))
  try(system(paste("dot -Tps", shQuote(paste0(ob,".dot")),
                   "-o", shQuote(paste0(ob,".ps"))), intern=TRUE), silent=TRUE)

  # ------------------------------------------------------------------
  # Parameter learning
  # ------------------------------------------------------------------
  t1    <- Sys.time()
  bnf   <- bn.fit(net, data=df_factor, method="mle",
                  replace.unidentifiable=TRUE)
  t_tot <- t_struct + as.numeric(Sys.time() - t1, units="secs")

  # Save .net and .pl
  try(write.net(paste0(ob,".net"), bnf), silent=TRUE)
  write_pl(bnf, paste0(ob, ".pl"))

  # ------------------------------------------------------------------
  # Per-fold log entry
  # ------------------------------------------------------------------
  log_line <- sprintf(
    paste0("Fold %d (row %d) | action=%s | state=%s | removed=%d | ",
           "train_n=%d | frac=%.2f | mmpc_edges=%d | connected=%s | ",
           "comps=%d | LC=%s | action_in_LC=%s | hc_arcs=%d | time=%.2fs | ",
           "alpha=%.2f | max.sx=%d | test=%s"),
    fi, fold_i,
    dt_unique$action[fold_i], cur_state_id,
    n_removed, train_sample_size, hc_fraction,
    sk_edges, sk_connected, sk_comps,
    lc_str, act_in_lc, final_arcs, t_tot,
    mmpc_alpha, mmpc_max_sx, mmpc_test)

  message("  ", log_line)
  write(log_line,
        file   = file.path(results_root, "training_numeralia.txt"),
        append = TRUE)

  # ------------------------------------------------------------------
  # Accumulate global summary
  # ------------------------------------------------------------------
  global_summary <- rbindlist(list(global_summary, data.table(
    Fold                  = fi,
    FoldIndex             = fold_i,
    Action                = dt_unique$action[fold_i],
    StateID               = cur_state_id,
    SamplesRemoved        = as.integer(n_removed),
    TrainSampleSize       = as.integer(train_sample_size),
    TrainingTime_s        = t_tot,
    MMPC_alpha            = mmpc_alpha,
    MMPC_max_sx           = as.integer(mmpc_max_sx),
    MMPC_test             = mmpc_test,
    MMPC_skeleton_edges   = as.integer(sk_edges),
    MMPC_sk_connected     = sk_connected,
    MMPC_sk_components    = as.integer(sk_comps),
    MMPC_LC_neighbors     = lc_str,
    MMPC_action_in_LC_nbr = act_in_lc,
    HC_final_arcs         = as.integer(final_arcs),
    HC_fraction           = hc_fraction
  )))

  rm(train_dt, train_sample, df_factor, bnf, net, net_mmpc); gc()
} # end fold loop

# =============================================================================
# Save global summary CSV and print final statistics
# =============================================================================
csv_path <- file.path(results_root, "global_summary.csv")
fwrite(global_summary, csv_path)
message("\nGlobal summary saved to: ", csv_path)

message("\n=== FINAL SUMMARY: ", combo_label, " ===")
message(sprintf("Folds completed       : %d", nrow(global_summary)))
message(sprintf("Avg training time     : %.2f +/- %.2f s",
                mean(global_summary$TrainingTime_s),
                sd(global_summary$TrainingTime_s)))
message(sprintf("Avg MMPC skeleton edges: %.1f +/- %.1f",
                mean(global_summary$MMPC_skeleton_edges),
                sd(global_summary$MMPC_skeleton_edges)))
message(sprintf("Folds with action in LC nbr: %d / %d (%.1f%%)",
                sum(global_summary$MMPC_action_in_LC_nbr),
                nrow(global_summary),
                100 * mean(global_summary$MMPC_action_in_LC_nbr)))
message(sprintf("Avg HC final arcs     : %.1f +/- %.1f",
                mean(global_summary$HC_final_arcs),
                sd(global_summary$HC_final_arcs)))

message("\nAction distribution across selected folds:")
print(global_summary[, .N, by=Action][order(Action)])

message("\nLC neighbor patterns (top 5):")
lc_freq <- sort(table(global_summary$MMPC_LC_neighbors), decreasing=TRUE)
for (j in seq_len(min(5L, length(lc_freq))))
  message("  ", names(lc_freq)[j], " : ", lc_freq[j], " folds (",
          round(100*lc_freq[j]/nrow(global_summary),1), "%)")

# Append summary to numeralia
summ_str <- sprintf(
  paste0("\n=== SUMMARY: %s ===\n",
         "Folds: %d\n",
         "Avg time: %.2f +/- %.2f s\n",
         "Avg MMPC edges: %.1f +/- %.1f\n",
         "Action in LC nbr: %d/%d (%.1f%%)\n",
         "Avg HC arcs: %.1f +/- %.1f\n"),
  combo_label,
  nrow(global_summary),
  mean(global_summary$TrainingTime_s), sd(global_summary$TrainingTime_s),
  mean(global_summary$MMPC_skeleton_edges), sd(global_summary$MMPC_skeleton_edges),
  sum(global_summary$MMPC_action_in_LC_nbr), nrow(global_summary),
  100*mean(global_summary$MMPC_action_in_LC_nbr),
  mean(global_summary$HC_final_arcs), sd(global_summary$HC_final_arcs))

write(summ_str,
      file   = file.path(results_root, "training_numeralia.txt"),
      append = TRUE)

message("\nScript finished.")
