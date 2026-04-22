#!/usr/bin/env Rscript
# =============================================================================
# CBNs_PC_sensitivity.R
#
# Sensitivity analysis over PC-stable hyperparameters.
# All combinations MUST use the same 78 stratified fold IDs and the same
# training subsample (25% by default) for valid cross-comparison.
#
# Fold selection (Option B1):
#   - On first run: draws 78 stratified folds (13 per action) from dt_unique
#     using FOLD_SEED, saves them to ./pc_sensitivity_folds/fold_ids.txt
#   - On subsequent runs: reads fold IDs from that file — no regeneration.
#
# Training subsampling:
#   - For each fold, a stratified subsample of the remaining dt is drawn
#     (default 25%) using TRAIN_SEED, reset at the start of every combination's
#     fold loop — guaranteeing identical subsamples across all 7 combinations.
#
# CPDAG -> DAG conversion uses a three-layer approach (from CBNs_LOOCV_training_PC.R):
#   Layer 1: cextend()           — standard CPDAG extension
#   Layer 2: HC on PC skeleton   — fallback if cextend fails
#   Layer 3: force-orient        — acyclic topological orientation, last resort
#
# Usage:
#   Rscript CBNs_PC_sensitivity.R <input_dir> \
#     --alpha 0.05 --max.sx 3 --test mi [--fraction 0.25]
#
# Parameters:
#   --alpha    : CI test significance threshold          (default 0.05)
#   --max.sx   : maximum conditioning set size; NULL=full PC (default 3)
#   --test     : CI test: mi, x2, mi-sh, etc.           (default mi)
#   --fraction : fraction of remaining data to use       (default 0.25)
#
# Parameter combinations:
#   Config 0 (original):  alpha=0.01, max.sx=NULL, test=mi  (replicates original experiment)
#   Config 1 (baseline):  alpha=0.05, max.sx=3,    test=mi
#   Config 2:             alpha=0.01, max.sx=3,    test=mi
#   Config 3:             alpha=0.10, max.sx=3,    test=mi
#   Config 4:             alpha=0.05, max.sx=2,    test=mi
#   Config 5:             alpha=0.05, max.sx=4,    test=mi
#   Config 6:             alpha=0.05, max.sx=NULL, test=mi  (full PC)
#   Config 7:             alpha=0.05, max.sx=3,    test=x2
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
FOLDS_DIR    <- "./pc_sensitivity_folds"
FOLDS_FILE   <- file.path(FOLDS_DIR, "fold_ids.txt")
TRAIN_SEED   <- 450L  # reset before each combination's fold loop
DEFAULT_FRAC <- 0.25  # default training fraction

# =============================================================================
# Argument parsing
# =============================================================================
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 1)
  stop("Usage: Rscript CBNs_PC_sensitivity.R <input_dir> ",
       "[--alpha V] [--max.sx V|NULL] [--test T] [--fraction F]")

input_dir   <- args[1]
pc_alpha    <- 0.05
pc_max_sx   <- 3L        # integer or NULL
pc_test     <- "mi"
hc_fraction <- DEFAULT_FRAC

named <- if (length(args) > 1) args[2:length(args)] else character(0)
i <- 1L
while (i <= length(named)) {
  fl <- named[i]
  if      (fl=="--alpha"    && i<length(named)){ pc_alpha    <- as.numeric(named[i+1]); i<-i+2L }
  else if (fl=="--max.sx"   && i<length(named)){
    pc_max_sx <- if (tolower(named[i+1]) %in% c("null","none","")) NULL
                 else as.integer(named[i+1])
    i<-i+2L
  }
  else if (fl=="--test"     && i<length(named)){ pc_test     <- named[i+1];             i<-i+2L }
  else if (fl=="--fraction" && i<length(named)){ hc_fraction <- as.numeric(named[i+1]); i<-i+2L }
  else { i<-i+1L }
}

valid_tests <- c("mi","mi-sh","x2","sp-mi","smc-mi")
if (!pc_test %in% valid_tests)
  stop("Invalid --test '", pc_test, "'. Valid: ", paste(valid_tests, collapse=", "))
if (hc_fraction <= 0 || hc_fraction > 1)
  stop("--fraction must be in (0, 1]")

# Build combo label for output directory
alpha_str  <- gsub("\\.", "p", sprintf("%.2f", pc_alpha))    # 0.05 -> 0p05
frac_str   <- gsub("0\\.", "p", sprintf("%.2f", hc_fraction)) # 0.25 -> p25
test_str   <- gsub("-", "_", pc_test)                          # mi-sh -> mi_sh
maxsx_str  <- if (is.null(pc_max_sx)) "NULL" else as.character(pc_max_sx)
combo_label <- sprintf("pc_%s_alpha%s_maxsx%s_frac%s",
                       test_str, alpha_str, maxsx_str, frac_str)

base_wd      <- getwd()
results_root <- file.path(base_wd, combo_label)
if (!dir.exists(results_root)) dir.create(results_root, recursive=TRUE)

message("=== PC-stable Sensitivity Configuration ===")
message("Input dir    : ", input_dir)
message("alpha        : ", pc_alpha)
message("max.sx       : ", ifelse(is.null(pc_max_sx), "NULL (full PC)", pc_max_sx))
message("test         : ", pc_test)
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
# CPDAG → DAG helpers (from CBNs_LOOCV_training_PC.R)
# =============================================================================

# Extract undirected skeleton edge pairs from adjacency matrix.
# Robust to internally inconsistent CPDAGs.
get_skeleton_pairs <- function(net) {
  a  <- amat(net)
  nd <- rownames(a)
  pairs <- list()
  for (r in seq_len(nrow(a)))
    for (cc in seq_len(ncol(a)))
      if (r < cc && (a[r,cc]==1 || a[cc,r]==1))
        pairs <- c(pairs, list(c(nd[r], nd[cc])))
  pairs
}

# Force a valid DAG from skeleton pairs via greedy topological orientation.
force_dag_from_skeleton <- function(skel_pairs, all_nodes) {
  node_order <- setNames(seq_along(all_nodes), all_nodes)
  arc_from <- character(0); arc_to <- character(0)
  for (p in skel_pairs) {
    if (node_order[p[1]] < node_order[p[2]]) {
      arc_from <- c(arc_from, p[1]); arc_to <- c(arc_to, p[2])
    } else {
      arc_from <- c(arc_from, p[2]); arc_to <- c(arc_to, p[1])
    }
  }
  if (length(arc_from) == 0) return(empty.graph(all_nodes))
  arc_df <- data.frame(from=arc_from, to=arc_to, stringsAsFactors=FALSE)
  tryCatch({
    g <- empty.graph(all_nodes)
    arcs(g, check.cycles=TRUE) <- arc_df
    g
  }, error=function(e) {
    g <- empty.graph(all_nodes)
    for (k in seq_len(nrow(arc_df))) {
      tryCatch({
        tmp <- g
        arcs(tmp, check.cycles=TRUE) <- rbind(arcs(g), arc_df[k,])
        g <<- tmp
      }, error=function(e) NULL)
    }
    g
  })
}

# =============================================================================
# Helper: write .pl probabilistic logic program from a fitted BN
# =============================================================================
write_pl <- function(bn_fit, path) {
  output_file <- file(path, "w")
  writeLines(c("%%%%%%%%%%%%%%%%%%%%%%%%%%",
               "% Exogenous variables",
               "%%%%%%%%%%%%%%%%%%%%%%%%%%\n"), con=output_file)
  u_idx <- 0L

  for (rv in bn_fit) {
    rv_name   <- rv$node; rv_values <- dimnames(rv$prob)[[1]]
    rv_numval <- length(rv_values)
    if (!identical(rv$parents, character(0))) next
    df2 <- as.data.frame(rv$prob); df2[] <- lapply(df2, as.character)
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
      ad_parts <- mapply(function(p,v) paste0(p,"::",u_name,"(",v,")"), probs, df2[,1])
      writeLines(paste0(paste(ad_parts, collapse="; "), "."), con=output_file)
      writeLines(paste0(rv_name, "(V) :- ", u_name, "(V).\n"), con=output_file)
    }
  }

  for (rv in bn_fit) {
    rv_name   <- rv$node; rv_values <- dimnames(rv$prob)[[1]]
    rv_numval <- length(rv_values)
    if (identical(rv$parents, character(0))) next
    df2 <- as.data.frame(rv$prob); df2[] <- lapply(df2, as.character)
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
          body <- paste0(body, ", ")
          if (identical(tolower(col_vals), c("false","true"))) {
            if (identical(tolower(df2[row_i,k]), "false"))
              body <- paste0(body, "\\+ ", names(df2)[k])
            else body <- paste0(body, names(df2)[k])
          } else body <- paste0(body, names(df2)[k], "(", df2[row_i,k], ")")
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
          body <- paste0(body, ", ")
          if (length(col_vals) > 2L)
            body <- paste0(body, names(df2)[k], "(", df2[row_i,k], ")")
          else if (identical(tolower(col_vals[1]),"false") &&
                   identical(tolower(col_vals[2]),"true")) {
            if (identical(tolower(df2[row_i,k]), "false"))
              body <- paste0(body, "\\+ ", names(df2)[k])
            else body <- paste0(body, names(df2)[k])
          } else body <- paste0(body, names(df2)[k], "(", df2[row_i,k], ")")
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
  Fold               = integer(),
  FoldIndex          = integer(),
  Action             = character(),
  StateID            = character(),
  SamplesRemoved     = integer(),
  TrainSampleSize    = integer(),
  TrainingTime_s     = numeric(),
  PC_alpha           = numeric(),
  PC_max_sx          = character(),  # character to accommodate "NULL"
  PC_test            = character(),
  PC_skeleton_edges  = integer(),
  PC_oriented_arcs   = integer(),
  PC_sk_connected    = logical(),
  PC_sk_components   = integer(),
  PC_cpdag_extended  = logical(),
  PC_orient_method   = character(),
  HC_final_arcs      = integer(),
  HC_fraction        = numeric()
)

# =============================================================================
# Main fold loop
# =============================================================================
set.seed(TRAIN_SEED)
message("\n=== Starting fold loop | seed reset to ", TRAIN_SEED, " ===")

action_list <- c("change_to_left","change_to_right","cruise",
                 "keep","swerve_left","swerve_right")

for (fi in seq_along(selected_folds)) {
  fold_i <- selected_folds[fi]
  message("\n--- Fold ", fi, "/", length(selected_folds),
          " (dt_unique row ", fold_i, ") | ", combo_label, " ---")

  fold_dir <- file.path(results_root, paste0("fold_", fi))
  if (!dir.exists(fold_dir)) dir.create(fold_dir, recursive=TRUE)

  # ------------------------------------------------------------------
  # Test dataset
  # ------------------------------------------------------------------
  cur_ex  <- dt_unique[fold_i,]
  dt_test <- cur_ex[rep(1:.N, each=6L)][, iaction := action_list]
  fwrite(dt_test[, lapply(.SD, as.character),
                 .SDcols=setdiff(names(dt_test),"state_id")],
         file.path(fold_dir, "test_data.csv"), quote=TRUE)

  # ------------------------------------------------------------------
  # Training dataset
  # ------------------------------------------------------------------
  cur_state_id <- dt_unique$state_id[fold_i]
  n_removed    <- sum(dt$state_id == cur_state_id)
  train_dt     <- dt[state_id != cur_state_id]

  if (cur_state_id %in% train_dt$state_id)
    stop("Data leakage: state_id found in training set at fold ", fi)

  message("  Removed ", n_removed, " rows matching state: ", cur_state_id)
  message("  Training rows available: ", nrow(train_dt))

  # ------------------------------------------------------------------
  # Stratified subsample
  # ------------------------------------------------------------------
  sample_size  <- max(1L, round(hc_fraction * nrow(train_dt)))
  act_unique   <- unique(train_dt$action)
  n_acts_train <- length(act_unique)

  if (sample_size < n_acts_train) {
    warning("sample_size (", sample_size, ") < number of actions (",
            n_acts_train, ") at fold ", fi)
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

  df_factor <- as.data.frame(lapply(
    train_sample[, lapply(.SD, as.character),
                 .SDcols=setdiff(names(train_sample),"state_id")],
    factor))

  # ------------------------------------------------------------------
  # PC-stable: learn CPDAG
  # ------------------------------------------------------------------
  t0 <- Sys.time()

  message("  Running pc.stable (test=", pc_test, ", alpha=", pc_alpha,
          ", max.sx=", ifelse(is.null(pc_max_sx),"NULL",pc_max_sx), ")...")

  cpdag <- pc.stable(df_factor,
                     test      = pc_test,
                     alpha     = pc_alpha,
                     max.sx    = pc_max_sx,
                     undirected = FALSE,
                     debug     = FALSE)

  # Skeleton diagnostics
  sk_edges <- nrow(skeleton(cpdag)$arcs) / 2
  cpdag_arcs <- arcs(cpdag)
  if (nrow(cpdag_arcs) > 0) {
    fwd    <- paste(cpdag_arcs[,1], cpdag_arcs[,2])
    rev    <- paste(cpdag_arcs[,2], cpdag_arcs[,1])
    or_arcs <- as.integer(sum(!fwd %in% rev))
  } else {
    or_arcs <- 0L
  }
  message("  PC skeleton edges : ", sk_edges)
  message("  PC oriented arcs  : ", or_arcs)

  # Skeleton connectivity
  adj <- amat(cpdag); nd <- rownames(adj); nn <- length(nd)
  vis <- rep(FALSE, nn); comps <- list()
  for (st in seq_len(nn)) {
    if (!vis[st]) {
      q <- st; comp <- integer(0)
      while (length(q) > 0) {
        u <- q[1]; q <- q[-1]
        if (!vis[u]) {
          vis[u] <- TRUE; comp <- c(comp, u)
          nb <- which(adj[u,]==1 | adj[,u]==1)
          q  <- unique(c(q, nb[!vis[nb]]))
        }
      }
      comps <- c(comps, list(nd[comp]))
    }
  }
  skel_conn  <- (length(comps) == 1L)
  skel_ncomp <- length(comps)
  message("  Skeleton connected: ", skel_conn, " (", skel_ncomp, " components)")

  # ------------------------------------------------------------------
  # CPDAG → DAG: three-layer approach
  # ------------------------------------------------------------------

  # Layer 1: cextend
  result <- tryCatch({
    dag <- cextend(cpdag, strict=FALSE)
    if (!acyclic(dag)) stop("cextend returned non-DAG")
    list(dag=dag, method="cextend")
  }, error=function(e) {
    message("  WARNING: cextend() failed: ", conditionMessage(e))
    NULL
  })

  # Layer 2: HC on PC skeleton
  if (is.null(result)) {
    result <- tryCatch({
      message("  Fallback layer 2: HC constrained to PC skeleton...")
      skel_pairs <- get_skeleton_pairs(cpdag)
      all_nd     <- nodes(cpdag)
      skel_set   <- lapply(skel_pairs, function(p) paste(sort(p), collapse="--"))
      bl_from <- character(0); bl_to <- character(0)
      for (n1 in all_nd) for (n2 in all_nd) {
        if (n1 != n2) {
          key <- paste(sort(c(n1,n2)), collapse="--")
          if (!key %in% skel_set) {
            bl_from <- c(bl_from, n1); bl_to <- c(bl_to, n2)
          }
        }
      }
      bl  <- if (length(bl_from) > 0)
               data.frame(from=bl_from, to=bl_to, stringsAsFactors=FALSE)
             else NULL
      dag <- hc(df_factor, blacklist=bl, score="bic", debug=FALSE,
                restart=0, perturb=1, max.iter=Inf, maxp=Inf, optimized=TRUE)
      if (!acyclic(dag)) stop("HC returned non-DAG")
      list(dag=dag, method="HC-on-skeleton")
    }, error=function(e) {
      message("  WARNING: HC fallback failed: ", conditionMessage(e))
      NULL
    })
  }

  # Layer 3: force-orient
  if (is.null(result)) {
    message("  Fallback layer 3: force-orienting skeleton acyclically...")
    skel_pairs <- get_skeleton_pairs(cpdag)
    dag        <- force_dag_from_skeleton(skel_pairs, nodes(cpdag))
    result     <- list(dag=dag, method="force-orient")
    message("  Force-orient produced ", nrow(arcs(dag)), " arcs.")
  }

  network_structure <- result$dag
  orient_method     <- result$method
  cpdag_ok          <- (orient_method == "cextend")
  final_arcs        <- nrow(arcs(network_structure))

  t_struct <- as.numeric(Sys.time() - t0, units="secs")
  message("  Orientation method: ", orient_method)
  message("  Final DAG arcs    : ", final_arcs,
          " | structure time: ", round(t_struct,2), "s")

  # Save per-fold skeleton summary
  sk_summary_file <- file.path(fold_dir, "pc_skeleton_summary.txt")
  cat(c(paste0("Fold ", fi, " (dt_unique row ", fold_i, ")"),
        paste0("PC test=", pc_test, "  alpha=", pc_alpha,
               "  max.sx=", ifelse(is.null(pc_max_sx),"NULL",pc_max_sx)),
        paste0("Skeleton edges  : ", sk_edges),
        paste0("Oriented arcs   : ", or_arcs),
        paste0("Connected       : ", skel_conn, "  Components: ", skel_ncomp),
        paste0("Orient method   : ", orient_method),
        paste0("CPDAG extended  : ", cpdag_ok),
        paste0("Final DAG arcs  : ", final_arcs),
        paste0("Struct time (s) : ", round(t_struct, 2)), ""),
      file=sk_summary_file, sep="\n")

  # Save .dot and .ps
  ob <- file.path(fold_dir, "cBN")
  write.dot(network_structure, file=paste0(ob, ".dot"))
  try(system(paste("dot -Tps", shQuote(paste0(ob,".dot")),
                   "-o", shQuote(paste0(ob,".ps"))), intern=TRUE), silent=TRUE)

  # ------------------------------------------------------------------
  # Parameter learning
  # ------------------------------------------------------------------
  t1    <- Sys.time()
  bnf   <- bn.fit(network_structure, data=df_factor, method="mle",
                  replace.unidentifiable=TRUE)
  t_tot <- t_struct + as.numeric(Sys.time() - t1, units="secs")

  try(write.net(paste0(ob,".net"), bnf), silent=TRUE)
  write_pl(bnf, paste0(ob, ".pl"))

  # ------------------------------------------------------------------
  # Per-fold log entry
  # ------------------------------------------------------------------
  log_line <- sprintf(
    paste0("Fold %d (row %d) | action=%s | state=%s | removed=%d | ",
           "train_n=%d | frac=%.2f | sk_edges=%d | or_arcs=%d | ",
           "connected=%s | comps=%d | method=%s | cpdag_ok=%s | ",
           "hc_arcs=%d | time=%.2fs | alpha=%.2f | max.sx=%s | test=%s"),
    fi, fold_i,
    dt_unique$action[fold_i], cur_state_id,
    n_removed, train_sample_size, hc_fraction,
    as.integer(sk_edges), or_arcs,
    skel_conn, skel_ncomp, orient_method, cpdag_ok,
    final_arcs, t_tot,
    pc_alpha, maxsx_str, pc_test)

  message("  ", log_line)
  write(log_line,
        file   = file.path(results_root, "training_numeralia.txt"),
        append = TRUE)

  # ------------------------------------------------------------------
  # Accumulate global summary
  # ------------------------------------------------------------------
  global_summary <- rbindlist(list(global_summary, data.table(
    Fold              = fi,
    FoldIndex         = fold_i,
    Action            = dt_unique$action[fold_i],
    StateID           = cur_state_id,
    SamplesRemoved    = as.integer(n_removed),
    TrainSampleSize   = as.integer(train_sample_size),
    TrainingTime_s    = t_tot,
    PC_alpha          = pc_alpha,
    PC_max_sx         = maxsx_str,
    PC_test           = pc_test,
    PC_skeleton_edges = as.integer(sk_edges),
    PC_oriented_arcs  = as.integer(or_arcs),
    PC_sk_connected   = skel_conn,
    PC_sk_components  = as.integer(skel_ncomp),
    PC_cpdag_extended = cpdag_ok,
    PC_orient_method  = orient_method,
    HC_final_arcs     = as.integer(final_arcs),
    HC_fraction       = hc_fraction
  )))

  rm(train_dt, train_sample, df_factor, bnf, network_structure, cpdag); gc()
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
message(sprintf("Avg PC skeleton edges : %.1f +/- %.1f",
                mean(global_summary$PC_skeleton_edges),
                sd(global_summary$PC_skeleton_edges)))
message(sprintf("Avg PC oriented arcs  : %.1f +/- %.1f",
                mean(global_summary$PC_oriented_arcs),
                sd(global_summary$PC_oriented_arcs)))
message(sprintf("CPDAG extended (Layer 1): %d / %d (%.1f%%)",
                sum(global_summary$PC_cpdag_extended),
                nrow(global_summary),
                100*mean(global_summary$PC_cpdag_extended)))
message(sprintf("Avg HC final arcs     : %.1f +/- %.1f",
                mean(global_summary$HC_final_arcs),
                sd(global_summary$HC_final_arcs)))

message("\nOrientation method distribution:")
print(table(global_summary$PC_orient_method))

message("\nAction distribution across selected folds:")
print(global_summary[, .N, by=Action][order(Action)])

summ_str <- sprintf(
  paste0("\n=== SUMMARY: %s ===\n",
         "Folds: %d\n",
         "Avg time: %.2f +/- %.2f s\n",
         "Avg PC skeleton edges: %.1f +/- %.1f\n",
         "Avg PC oriented arcs: %.1f +/- %.1f\n",
         "CPDAG extended: %d/%d (%.1f%%)\n",
         "Avg HC arcs: %.1f +/- %.1f\n"),
  combo_label,
  nrow(global_summary),
  mean(global_summary$TrainingTime_s), sd(global_summary$TrainingTime_s),
  mean(global_summary$PC_skeleton_edges), sd(global_summary$PC_skeleton_edges),
  mean(global_summary$PC_oriented_arcs),  sd(global_summary$PC_oriented_arcs),
  sum(global_summary$PC_cpdag_extended),  nrow(global_summary),
  100*mean(global_summary$PC_cpdag_extended),
  mean(global_summary$HC_final_arcs), sd(global_summary$HC_final_arcs))

write(summ_str,
      file   = file.path(results_root, "training_numeralia.txt"),
      append = TRUE)

message("\nScript finished.")
