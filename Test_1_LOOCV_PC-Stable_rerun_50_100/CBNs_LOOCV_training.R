#!/usr/bin/env Rscript
# =============================================================================
# CBNs_LOOCV_training_PC.R
#
# Modified from CBNs_LOOCV_training.R to use PC-stable (constraint-based)
# instead of HC+BIC (score-based) for structural learning.
#
# Key algorithmic changes:
#   - pc.stable() replaces hc()
#   - No whitelist or blacklist (fully unconstrained)
#   - PC returns a CPDAG; we convert to a DAG via cextend() before bn.fit()
#   - Recommended parameters for large discrete datasets (see below)
#
# PC-stable parameters used:
#   test   = "mi"   — mutual information CI test for discrete data
#   alpha  = 0.01   — tighter than default 0.05; justified by very large N
#                     (at ~1M rows, standard alpha finds spurious edges easily)
#   max.sx = NULL   — no limit on conditioning set size (fully unconstrained)
#   undirected = FALSE — attempt v-structure and Meek orientation
#
# Why alpha=0.01 and not 0.05?
#   With ~1-2M training rows the MI test has extreme statistical power.
#   At alpha=0.05 even tiny, practically irrelevant dependencies become
#   "significant", producing dense graphs. alpha=0.01 (or 0.001) keeps
#   the skeleton sparser and closer to the true structure.
#   This is a well-known large-N issue with constraint-based methods.
#
# Usage (same interface as original):
#   Rscript CBNs_LOOCV_training_PC.R <input_dir> [reps] [percentages]
#
# Examples:
#   Rscript CBNs_LOOCV_training_PC.R ./Shared_CSVs 1 01
#   Rscript CBNs_LOOCV_training_PC.R ./Shared_CSVs 1 01,50,100
#   Rscript CBNs_LOOCV_training_PC.R ./Shared_CSVs 1-5 01,25,50,75,90
# =============================================================================

suppressMessages({
  for (pkg in c("data.table","conflicted","bnlearn",
                "dplyr","stringr","tidyr","tidyverse","arules")) {
    if (!requireNamespace(pkg, quietly = TRUE))
      install.packages(pkg, repos = "https://cloud.r-project.org")
  }
})

library(data.table)
library(conflicted)
conflict_prefer("setdiff", "base")
library(bnlearn)
library(dplyr); library(stringr); library(tidyr); library(tidyverse); library(arules)

dec_round <- 7

# =============================================================================
# PC-stable hyperparameters — change here to experiment
# =============================================================================
PC_TEST  <- "mi"    # CI test: "mi" (mutual information) for discrete data
PC_ALPHA <- 0.01    # significance threshold; tighter = sparser graph
                    # recommended range for large N: 0.001 to 0.01
PC_MAXSX <- NULL    # max conditioning set size; NULL = unconstrained (true PC)
                    # set to e.g. 3 to speed up at cost of completeness

# =============================================================================
# Argument parsing (identical to original)
# =============================================================================
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop(paste(
    "Usage:",
    "  Rscript CBNs_LOOCV_training_PC.R <input_dir> [reps] [percentages]",
    "Examples:",
    "  Rscript CBNs_LOOCV_training_PC.R ./Shared_CSVs 1 01",
    "  Rscript CBNs_LOOCV_training_PC.R ./Shared_CSVs 1-5 01,25,50,75,90",
    sep = "\n"
  ))
}

input_subdir1 <- args[1]
reps_str        <- "1-5"
percentages_str <- "01,25,50,75,90"

if (length(args) >= 2) {
  if (grepl(",", args[2]) || grepl("-", args[2])) {
    reps_str <- args[2]
    if (length(args) >= 3) percentages_str <- args[3]
  } else {
    reps_str <- paste0("1-", args[2])
    if (length(args) >= 3) percentages_str <- args[3]
  }
}

if (grepl("-", reps_str)) {
  bounds <- as.integer(strsplit(reps_str, "-")[[1]]); reps <- seq(bounds[1], bounds[2])
} else if (grepl(",", reps_str)) {
  reps <- as.integer(strsplit(reps_str, ",")[[1]])
} else {
  reps <- as.integer(reps_str)
}

percentages <- sprintf("%02d", as.integer(strsplit(percentages_str, ",")[[1]] |> trimws()))

message("=== PC-stable LOOCV Training ===")
message("Input directory : ", input_subdir1)
message("Repetitions     : ", paste(reps, collapse=", "))
message("Percentages     : ", paste(percentages, collapse=", "))
message("CI test         : ", PC_TEST)
message("Alpha           : ", PC_ALPHA)
message("Max cond. set   : ", ifelse(is.null(PC_MAXSX), "unlimited (true PC)", PC_MAXSX))

# =============================================================================
# Configuration
# =============================================================================
seeds           <- c(300, 456, 211, 26, 500, 1001, 724, 881, 91, 255)
shared_csv_path <- "./Shared_CSVs"
base_wd         <- getwd()
message("Working directory: ", base_wd)

# =============================================================================
# Utility functions
# =============================================================================
adjust_values <- function(freq) {
  freq[is.na(freq)] <- 0.0
  freq[freq <= 0.0001] <- 0.001
  total_sum <- sum(freq)
  if (total_sum == 0.0) freq <- rep(1/length(freq), length(freq))
  else if (total_sum > 1) freq <- freq / total_sum
  return(freq)
}

report_progress <- function(step_msg, progress_file = NULL) {
  message(step_msg)
  if (!is.null(progress_file)) cat(step_msg, "\n", file = progress_file, append = TRUE)
}

# =============================================================================
# Input validation
# =============================================================================
for (f in file.path(shared_csv_path, c("complete_DB_discrete.csv","crashes.csv","no_crashes.csv")))
  if (!file.exists(f)) stop("Input file missing: ", f)

# =============================================================================
# Read datasets
# =============================================================================
dt            <- fread(file.path(input_subdir1,"complete_DB_discrete.csv"), colClasses="character")
dt            <- dt[complete.cases(dt)]
dt_crashes    <- fread(file.path(input_subdir1,"crashes.csv"),    colClasses="character")
dt_no_crashes <- fread(file.path(input_subdir1,"no_crashes.csv"), colClasses="character")
dt_crashes[,    orig_label_lc := "True"]
dt_no_crashes[, orig_label_lc := "False"]

dt_unique <- rbindlist(list(dt_crashes, dt_no_crashes))
dt_unique[, latent_collision := "True"]

state_cols <- c("curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W")
dt[,        state_id := do.call(paste, c(.SD, sep="_")), .SDcols=state_cols]
dt_unique[, state_id := do.call(paste, c(.SD, sep="_")), .SDcols=state_cols]

for (nm in c("dt","dt_unique"))
  if (anyNA(get(nm)$state_id)) stop("NA values in state_id in ", nm)

# =============================================================================
# Global summary
# =============================================================================
global_summary <- data.table(
  Repetition      = integer(),
  Percentage      = character(),
  Fold            = integer(),
  TrainingTime_s  = numeric(),
  SamplesRemoved  = integer(),
  TrainSampleSize = integer(),
  Skeleton_edges    = integer(),
  Oriented_arcs     = integer(),
  Orient_method     = character(),
  CPDAG_extended    = logical(),
  Skeleton_connected= logical(),
  Skeleton_components= integer()
)

# =============================================================================
# Main loops
# =============================================================================
for (r_idx in seq_along(reps)) {
  rep_num <- reps[r_idx]
  seed    <- seeds[r_idx]
  set.seed(seed)

  message("\n==============================")
  message("Starting repetition: ", rep_num)
  message("==============================\n")

  rep_dir      <- file.path(base_wd, sprintf("rep_%d", rep_num))
  rep_test_dir <- file.path(rep_dir, "test_data")
  if (!dir.exists(rep_dir))      dir.create(rep_dir,      recursive=TRUE)
  if (!dir.exists(rep_test_dir)) dir.create(rep_test_dir, recursive=TRUE)

  dt_local        <- copy(dt)
  dt_crashes_l    <- copy(dt_crashes)
  dt_no_crashes_l <- copy(dt_no_crashes)
  dt_unique_l     <- copy(dt_unique)

  n_folds <- nrow(dt_unique_l)
  message("LOOCV folds: ", n_folds)

  for (percentage in percentages) {
    message("\n--- Repetition ", rep_num, " | Percentage ", percentage, " ---\n")

    pct_dir      <- file.path(rep_dir, percentage)
    training_dir <- file.path(pct_dir, "training_data")
    cbns_dir     <- file.path(pct_dir, "cBNs")
    for (d in c(training_dir, cbns_dir)) if (!dir.exists(d)) dir.create(d, recursive=TRUE)

    fraction <- as.numeric(percentage) / 100
    if (fraction <= 0 | fraction > 1) stop("Percentage must be 1-100")

    training_times     <- numeric()
    samples_removed    <- numeric()
    train_sample_sizes <- numeric()
    skeleton_edges_v   <- integer()
    oriented_arcs_v    <- integer()
    cpdag_extended_v   <- logical()
    orient_method_v    <- character()
    skel_connected_v   <- logical()
    skel_components_v  <- integer()

    for (i in seq_len(n_folds)) {
      message("Processing fold ", i, " of ", n_folds)

      # --- Test dataset ---
      current_test_example <- dt_unique_l[i, ]
      report_progress("Creating test dataset...")
      action_list <- c("change_to_left","change_to_right","cruise","keep","swerve_left","swerve_right")

      # FIX: assign iaction directly — after rep(1:.N, each=6) the table has
      # exactly 6 rows and action_list has exactly 6 elements.
      dt_test <- current_test_example[rep(1:.N, each=6)]
      dt_test[, iaction := action_list]

      message("dt_test dims: ", paste(dim(dt_test), collapse=" x "))
      if (anyNA(dt_test$state_id)) stop("NA in state_id in dt_test")

      test_file <- file.path(rep_test_dir, paste0("test_fold_", i, ".csv"))
      fwrite(dt_test[, lapply(.SD, as.character), .SDcols=setdiff(names(dt_test),"state_id")],
             test_file, quote=TRUE)
      report_progress(paste("Test data saved:", test_file))

      # --- Training dataset ---
      current_state_id <- dt_unique_l$state_id[i]
      num_removed      <- sum(dt_local$state_id == current_state_id)
      samples_removed  <- c(samples_removed, num_removed)
      train_dt         <- dt_local[state_id != current_state_id]

      if (current_state_id %in% train_dt$state_id)
        stop("Data leakage in fold ", i)

      # Stratified sample
      action_list_unique <- unique(train_dt$action)
      sample_size        <- round(fraction * nrow(train_dt))
      if (sample_size < length(action_list_unique)) {
        message("Warning: sample_size (", sample_size, ") < n_actions (", length(action_list_unique), ")")
        train_sample <- train_dt[sample(.N, min(sample_size,.N))]
      } else {
        min_samples    <- train_dt[, .SD[sample(.N, min(1,.N))], by=action]
        remaining_size <- sample_size - nrow(min_samples)
        train_sample   <- if (remaining_size > 0)
          rbindlist(list(min_samples, train_dt[sample(.N, remaining_size)]))
        else min_samples
      }

      train_sample_size  <- nrow(train_sample)
      train_sample_sizes <- c(train_sample_sizes, train_sample_size)
      message("Train sample size: ", train_sample_size)

      action_dist <- train_sample[, .N, by=action]
      message("Action dist:\n", paste(capture.output(print(action_dist)), collapse="\n"))

      train_sample_to_save <- train_sample[, lapply(.SD, as.character),
                                           .SDcols=setdiff(names(train_sample),"state_id")]
      df_factor <- as.data.frame(lapply(train_sample_to_save, factor))

      # =========================================================================
      # PC-stable structural learning (replaces hc)
      # =========================================================================
      start_time_struct <- Sys.time()

      message("Running PC-stable (test=", PC_TEST, ", alpha=", PC_ALPHA,
              ", max.sx=", ifelse(is.null(PC_MAXSX),"NULL",PC_MAXSX), ")...")

      # pc.stable returns a CPDAG (partially directed acyclic graph).
      # It may contain undirected edges where orientation cannot be determined.
      cpdag <- pc.stable(
        df_factor,
        test      = PC_TEST,
        alpha     = PC_ALPHA,
        max.sx    = PC_MAXSX,   # NULL = unconstrained (true PC algorithm)
        undirected = FALSE,     # attempt orientation via v-structures + Meek rules
        debug     = FALSE
      )

      # Skeleton diagnostics
      sk_edges <- nrow(skeleton(cpdag)$arcs) / 2  # skeleton arcs are undirected, counted twice
      # Count truly directed arcs: A→B is directed only if B→A is NOT also present
      # (bnlearn represents undirected edges as two opposing arcs, so nrow(arcs())
      #  double-counts them — this gives the correct oriented-only count)
      cpdag_arcs <- arcs(cpdag)
      if (nrow(cpdag_arcs) > 0) {
        fwd <- paste(cpdag_arcs[,1], cpdag_arcs[,2])
        rev <- paste(cpdag_arcs[,2], cpdag_arcs[,1])
        or_arcs <- as.integer(sum(!fwd %in% rev))
      } else {
        or_arcs <- 0L
      }
      message("PC skeleton edges  : ", sk_edges)
      message("PC oriented arcs   : ", or_arcs)

      # Skeleton connectivity
      adj    <- amat(cpdag); nd <- rownames(adj); nn <- length(nd)
      vis    <- rep(FALSE, nn); comps <- list()
      for (st in seq_len(nn)) {
        if (!vis[st]) {
          q <- st; comp <- integer(0)
          while (length(q)>0) {
            u <- q[1]; q <- q[-1]
            if (!vis[u]) {
              vis[u] <- TRUE; comp <- c(comp, u)
              nb <- which(adj[u,]==1 | adj[,u]==1)  # undirected: check both directions
              q  <- unique(c(q, nb[!vis[nb]]))
            }
          }
          comps <- c(comps, list(nd[comp]))
        }
      }
      skel_conn <- (length(comps)==1); skel_ncomp <- length(comps)
      message("Skeleton connected : ", skel_conn, " (", skel_ncomp, " components)")

      skeleton_edges_v  <- c(skeleton_edges_v,  as.integer(sk_edges))
      oriented_arcs_v   <- c(oriented_arcs_v,   as.integer(or_arcs))
      skel_connected_v  <- c(skel_connected_v,  skel_conn)
      skel_components_v <- c(skel_components_v, skel_ncomp)

      # -----------------------------------------------------------------------
      # Convert CPDAG → DAG for bn.fit
      #
      # Three-layer approach to handle inconsistent CPDAGs robustly:
      #
      # Layer 1: cextend() — standard CPDAG extension (may fail if PC
      #          produced conflicting v-structure orientations)
      #
      # Layer 2: HC constrained to PC skeleton — extract skeleton directly
      #          from the adjacency matrix (NOT via skeleton(cpdag), which
      #          can itself fail on inconsistent CPDAGs), build a blacklist
      #          of non-skeleton edges, let HC orient freely within it
      #
      # Layer 3: Force-orient — if HC also fails, take the symmetric part
      #          of the adjacency matrix (undirected skeleton) and orient
      #          every edge arbitrarily but acyclically using a topological
      #          sort. Guaranteed to produce a valid DAG.
      # -----------------------------------------------------------------------

      # Helper: extract undirected skeleton edges from raw adjacency matrix.
      # Works even when the CPDAG is internally inconsistent.
      get_skeleton_pairs <- function(net) {
        a  <- amat(net)
        nd <- rownames(a)
        pairs <- list()
        for (r in seq_len(nrow(a))) {
          for (cc in seq_len(ncol(a))) {
            if (r < cc && (a[r,cc]==1 || a[cc,r]==1))
              pairs <- c(pairs, list(c(nd[r], nd[cc])))
          }
        }
        pairs
      }

      # Helper: force a DAG from a skeleton pair list via greedy topological
      # orientation — orients each undirected edge low-index → high-index,
      # which is acyclic by construction on a node-indexed ordering.
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
        if (length(arc_from)==0)
          return(empty.graph(all_nodes))
        arc_df <- data.frame(from=arc_from, to=arc_to, stringsAsFactors=FALSE)
        tryCatch({
          g <- empty.graph(all_nodes)
          arcs(g, check.cycles=TRUE) <- arc_df
          g
        }, error=function(e) {
          # Cycle introduced — remove offending arcs one by one
          g <- empty.graph(all_nodes)
          for (k in seq_len(nrow(arc_df))) {
            tryCatch(
              { tmp <- g; arcs(tmp, check.cycles=TRUE) <- rbind(arcs(g), arc_df[k,]); g <- tmp },
              error=function(e) NULL
            )
          }
          g
        })
      }

      # ---- Layer 1: cextend ------------------------------------------------
      result <- tryCatch({
        dag <- cextend(cpdag, strict=FALSE)
        # Verify it is actually a DAG (no undirected edges)
        if (!acyclic(dag)) stop("cextend returned non-DAG")
        list(dag=dag, method="cextend")
      }, error=function(e) {
        message("WARNING: cextend() failed: ", conditionMessage(e))
        NULL
      })

      # ---- Layer 2: HC on PC skeleton --------------------------------------
      if (is.null(result)) {
        result <- tryCatch({
          message("Fallback layer 2: HC constrained to PC skeleton...")
          skel_pairs <- get_skeleton_pairs(cpdag)
          all_nd     <- nodes(cpdag)

          # Edges IN skeleton (both directions allowed for HC)
          skel_set <- lapply(skel_pairs, function(p) paste(sort(p), collapse="--"))

          # Blacklist: directed arcs whose undirected version is NOT in skeleton
          bl_from <- character(0); bl_to <- character(0)
          for (n1 in all_nd) for (n2 in all_nd) {
            if (n1 != n2) {
              key <- paste(sort(c(n1,n2)), collapse="--")
              if (!key %in% skel_set) {
                bl_from <- c(bl_from, n1); bl_to <- c(bl_to, n2)
              }
            }
          }
          bl <- if (length(bl_from)>0)
            data.frame(from=bl_from, to=bl_to, stringsAsFactors=FALSE) else NULL

          dag <- hc(df_factor, blacklist=bl, score="bic", debug=FALSE,
                    restart=0, perturb=1, max.iter=Inf, maxp=Inf, optimized=TRUE)
          if (!acyclic(dag)) stop("HC returned non-DAG")
          list(dag=dag, method="HC-on-skeleton")
        }, error=function(e) {
          message("WARNING: HC fallback failed: ", conditionMessage(e))
          NULL
        })
      }

      # ---- Layer 3: Force-orient skeleton ----------------------------------
      if (is.null(result)) {
        message("Fallback layer 3: force-orienting skeleton acyclically...")
        skel_pairs <- get_skeleton_pairs(cpdag)
        dag        <- force_dag_from_skeleton(skel_pairs, nodes(cpdag))
        result     <- list(dag=dag, method="force-orient")
        message("Force-orient produced ", nrow(arcs(dag)), " arcs.")
      }

      network_structure <- result$dag
      orient_method     <- result$method
      cpdag_ok          <- (orient_method == "cextend")
      cpdag_extended_v  <- c(cpdag_extended_v, cpdag_ok)
      orient_method_v   <- c(orient_method_v,  orient_method)

      message("Orientation method : ", orient_method)
      message("Final DAG arcs     : ", nrow(arcs(network_structure)))

      end_time_struct     <- Sys.time()
      elapsed_time_struct <- as.numeric(end_time_struct - start_time_struct, units="secs")

      # --- Save skeleton summary ---
      skel_summary_file <- file.path(cbns_dir, "skeleton_summary.txt")
      cat(c(
        paste0("Fold ", i, " -------"),
        paste0("test=",PC_TEST,"  alpha=",PC_ALPHA,"  max.sx=",
               ifelse(is.null(PC_MAXSX),"NULL",PC_MAXSX)),
        paste0("Skeleton edges  : ", sk_edges),
        paste0("Oriented arcs   : ", or_arcs),
        paste0("Connected       : ", skel_conn, "  Components: ", skel_ncomp),
        paste0("Orient method   : ", orient_method),
        paste0("Final DAG arcs  : ", nrow(arcs(network_structure))),
        paste0("Struct time (s) : ", round(elapsed_time_struct, 2)),
        ""
      ), file=skel_summary_file, sep="\n", append=TRUE)

      # --- Save .dot and .ps ---
      output_base <- file.path(cbns_dir, paste0("cBN_", i))
      model_dot   <- paste0(output_base, ".dot")
      write.dot(network_structure, file=model_dot)
      try({
        system(paste("dot -Tps", shQuote(model_dot), "-o",
                     shQuote(paste0(output_base,".ps"))), intern=TRUE)
      }, silent=TRUE)

      # --- Parameter learning ---
      start_time_param  <- Sys.time()
      bn_fit            <- bn.fit(network_structure, data=df_factor,
                                  method="mle", replace.unidentifiable=TRUE)
      elapsed_time_param<- as.numeric(Sys.time()-start_time_param, units="secs")

      total_training_time <- elapsed_time_struct + elapsed_time_param
      training_times      <- c(training_times, total_training_time)

      # --- Numeralia ---
      numeralia_file <- file.path(cbns_dir, "training_numeralia.txt")
      numeralia_line <- sprintf(
        "Fold %d: Time=%.2fs | Removed=%d | TrainSize=%d | SkelEdges=%d | OrientedArcs=%d | Method=%s",
        i, total_training_time, num_removed, train_sample_size,
        as.integer(sk_edges), or_arcs, orient_method
      )
      if (!file.exists(numeralia_file)) writeLines(numeralia_line, numeralia_file)
      else write(sprintf("\n%s", numeralia_line), file=numeralia_file, append=TRUE)

      # --- .net file ---
      try(write.net(paste0(output_base,".net"), bn_fit), silent=TRUE)

      # --- Write .pl ---
      output_pl   <- paste0(output_base, ".pl")
      message("Writing .pl to: ", output_pl)
      output_file <- file(output_pl, "w")
      writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%", con=output_file)
      writeLines("% Exogenous variables",      con=output_file)
      writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n", con=output_file)

      u_idx <- 0
      # Root nodes
      for (rv in bn_fit) {
        rv_name   <- rv$node
        rv_values <- dimnames(rv$prob)[[1]]
        rv_numval <- length(rv_values)
        if (!identical(rv$parents, character(0))) next

        df2 <- as.data.frame(rv$prob); df2[] <- lapply(df2, as.character)
        df2$Freq <- adjust_values(as.numeric(df2$Freq))

        if (rv_numval==2 && identical(tolower(rv_values), c("false","true"))) {
          u_idx <- u_idx+1
          writeLines(paste0(format(round(df2[2,"Freq"],dec_round),nsmall=dec_round,scientific=FALSE),
                            "::u",u_idx,".\n",rv_name," :- u",u_idx,".\n"), con=output_file)
        } else {
          u_name  <- paste0("u_",rv_name)
          probs   <- sapply(seq_len(rv_numval),function(j)
            format(round(df2[j,"Freq"],dec_round),nsmall=dec_round,scientific=FALSE))
          ad_line <- paste(mapply(function(p,v)paste0(p,"::",u_name,"(",v,")"),
                                  probs,df2[,1]),collapse="; ")
          writeLines(paste0(ad_line,"."),                        con=output_file)
          writeLines(paste0(rv_name,"(V) :- ",u_name,"(V).\n"), con=output_file)
        }
      }

      # Non-root nodes
      for (rv in bn_fit) {
        rv_name   <- rv$node
        rv_values <- dimnames(rv$prob)[[1]]
        rv_numval <- length(rv_values)
        if (identical(rv$parents, character(0))) next

        df2 <- as.data.frame(rv$prob); df2[] <- lapply(df2, as.character)
        df2$Freq <- as.numeric(df2$Freq)

        for (ir in seq(1, nrow(df2), by=rv_numval)) {
          df2$Freq[ir:(ir+rv_numval-1)] <- adjust_values(df2$Freq[ir:(ir+rv_numval-1)])

          if (rv_numval==2 && identical(tolower(rv_values), c("false","true"))) {
            u_idx <- u_idx+1
            hd <- paste0(format(round(df2[ir+1,"Freq"],dec_round),nsmall=dec_round,scientific=FALSE),
                         "::u",u_idx,".")
            bd <- paste0(rv_name," :- u",u_idx)
            for (k in seq(2, ncol(df2)-1)) {
              cv <- unique(as.character(unlist(df2[,k]))); bd <- paste0(bd,", ")
              if (identical(tolower(cv), c("false","true"))) {
                if (identical(tolower(df2[ir,k]),"false")) bd <- paste0(bd,"\\+ ",names(df2)[k])
                else bd <- paste0(bd,names(df2)[k])
              } else bd <- paste0(bd,names(df2)[k],"(",df2[ir,k],")")
            }
            writeLines(paste0(hd,"\n",bd,".\n"), con=output_file)

          } else {
            u_idx   <- u_idx+1
            u_name  <- paste0("u",u_idx)
            probs   <- sapply(ir:(ir+rv_numval-1),function(r)
              format(round(df2[r,"Freq"],dec_round),nsmall=dec_round,scientific=FALSE))
            ad_line <- paste(mapply(function(p,v)paste0(p,"::",u_name,"(",v,")"),
                                    probs,rv_values),collapse="; ")
            writeLines(paste0(ad_line,"."), con=output_file)
            bd <- paste0(rv_name,"(V) :- ",u_name,"(V)")
            for (k in seq(2, ncol(df2)-1)) {
              cv <- unique(as.character(unlist(df2[,k]))); bd <- paste0(bd,", ")
              if (length(cv)>2) bd <- paste0(bd,names(df2)[k],"(",df2[ir,k],")")
              else if (identical(tolower(cv[1]),"false") && identical(tolower(cv[2]),"true")) {
                if (identical(tolower(df2[ir,k]),"false")) bd <- paste0(bd,"\\+ ",names(df2)[k])
                else bd <- paste0(bd,names(df2)[k])
              } else bd <- paste0(bd,names(df2)[k],"(",df2[ir,k],")")
            }
            writeLines(paste0(bd,".\n"), con=output_file)
          }
        }
      }
      close(output_file)

      rm(train_dt, train_sample, df_factor, bn_fit, network_structure, cpdag)
      gc()
    } # end fold loop

    # =========================================================================
    # Populate global summary (BUG FIX: inside percentage loop)
    # =========================================================================
    if (length(training_times) > 0) {
      global_summary <- rbindlist(list(global_summary, data.table(
        Repetition       = rep(rep_num,          length(training_times)),
        Percentage       = rep(percentage,        length(training_times)),
        Fold             = seq_len(n_folds),
        TrainingTime_s   = training_times,
        SamplesRemoved   = as.integer(samples_removed),
        TrainSampleSize  = as.integer(train_sample_sizes),
        Skeleton_edges   = skeleton_edges_v,
        Oriented_arcs    = oriented_arcs_v,
        Orient_method    = orient_method_v,
        CPDAG_extended   = cpdag_extended_v,
        Skeleton_connected  = skel_connected_v,
        Skeleton_components = skel_components_v
      )))
    }

    # Summary numeralia
    numeralia_file <- file.path(cbns_dir, "training_numeralia.txt")
    if (length(training_times) > 0) {
      n_ext   <- sum(cpdag_extended_v)
      n_conn  <- sum(skel_connected_v)
      meth_tbl <- table(orient_method_v)
      meth_str <- paste(names(meth_tbl), meth_tbl, sep=":", collapse=" | ")
      summary_str <- sprintf(
        "\nSummary (test=%s, alpha=%.3f, max.sx=%s):\n  Avg Training Time  = %.2f s (SD=%.2f)\n  Avg Removed        = %.1f (SD=%.1f)\n  Avg Train Size     = %.1f (SD=%.1f)\n  Avg Skeleton Edges = %.1f (SD=%.1f)\n  Avg Oriented Arcs  = %.1f (SD=%.1f)\n  Orientation method = %s\n  Skeleton connected = %d/%d folds\n",
        PC_TEST, PC_ALPHA, ifelse(is.null(PC_MAXSX),"NULL",PC_MAXSX),
        mean(training_times),     sd(training_times),
        mean(samples_removed),    sd(samples_removed),
        mean(train_sample_sizes), sd(train_sample_sizes),
        mean(skeleton_edges_v),   sd(skeleton_edges_v),
        mean(oriented_arcs_v),    sd(oriented_arcs_v),
        meth_str,
        n_conn, length(skel_connected_v)
      )
      write(summary_str, file=numeralia_file, append=TRUE)
      message(summary_str)
    }

    message("Completed percentage ", percentage, " for repetition ", rep_num)
  } # end percentage loop

  # Per-rep summary
  rep_summ <- global_summary[Repetition == rep_num]
  if (nrow(rep_summ) > 0) {
    message("\nRepetition ", rep_num, " summary:")
    message("  Avg training time (s)  : ", round(mean(rep_summ$TrainingTime_s),2),
            " ± ", round(sd(rep_summ$TrainingTime_s),2))
    message("  Avg samples removed    : ", round(mean(rep_summ$SamplesRemoved),1),
            " ± ", round(sd(rep_summ$SamplesRemoved),1))
    message("  Avg train sample size  : ", round(mean(rep_summ$TrainSampleSize),1),
            " ± ", round(sd(rep_summ$TrainSampleSize),1))
    message("  Avg skeleton edges     : ", round(mean(rep_summ$Skeleton_edges),1),
            " ± ", round(sd(rep_summ$Skeleton_edges),1))
    message("  CPDAG extended cleanly : ",
            sum(rep_summ$CPDAG_extended), "/", nrow(rep_summ), " folds")
  }
  message("\nFinished repetition: ", rep_num, "\n")

} # end rep loop

# =============================================================================
# Save global summary
# =============================================================================
summary_file <- file.path(base_wd, "global_training_summary_PC.csv")
fwrite(global_summary, summary_file)
message("\nGlobal summary: ", summary_file)
message("Script finished successfully.")
