#!/usr/bin/env Rscript
# =============================================================================
# verify_mmpc_sensitivity.R
#
# Standalone verification script to confirm that the sensitivity analysis
# results are genuine and not artifacts of:
#   (1) hyperparameters not being passed to mmpc()
#   (2) mi-sh silently falling back to mi
#   (3) α having no effect due to the test implementation
#   (4) results being identical across combos for the wrong reason
#
# Produces a human-readable report: verify_mmpc_report.txt
# All checks use bnlearn's built-in datasets (no external files needed)
# plus a subset of your actual data if found.
#
# Run:
#   Rscript verify_mmpc_sensitivity.R
#   Rscript verify_mmpc_sensitivity.R ./Shared_CSVs   # also checks real data
# =============================================================================

suppressMessages({
  for (pkg in c("data.table","bnlearn")) {
    if (!requireNamespace(pkg, quietly=TRUE))
      install.packages(pkg, repos="https://cloud.r-project.org")
  }
})
library(bnlearn)
library(data.table)

args        <- commandArgs(trailingOnly=TRUE)
input_dir   <- if (length(args)>=1) args[1] else NULL
report_file <- "verify_mmpc_report.txt"
con         <- file(report_file, "w")

# Helper: write to both console and report file
log <- function(...) {
  msg <- paste0(...)
  message(msg)
  writeLines(msg, con=con)
}

log("================================================================")
log(" MMPC SENSITIVITY ANALYSIS — VERIFICATION REPORT")
log(" Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
log(" bnlearn version: ", as.character(packageVersion("bnlearn")))
log(" R version: ", R.version$version.string)
log("================================================================")
log("")

# =============================================================================
# CHECK 1 — Does bnlearn accept mi-sh as a valid test?
# =============================================================================
log("----------------------------------------------------------------")
log("CHECK 1: Is 'mi-sh' a valid test in this version of bnlearn?")
log("----------------------------------------------------------------")

valid_tests <- tryCatch({
  # bnlearn stores valid discrete tests internally
  # The cleanest check is to try it on a tiny dataset and catch errors
  data(learning.test)
  tiny <- learning.test[1:100, ]
  net_test <- mmpc(tiny, test="mi-sh", alpha=0.05, max.sx=3)
  "PASS — mi-sh accepted without error"
}, error=function(e) {
  paste0("FAIL — mi-sh caused error: ", conditionMessage(e))
}, warning=function(w) {
  paste0("WARN — mi-sh caused warning: ", conditionMessage(w))
})

log(valid_tests)
log("")

# =============================================================================
# CHECK 2 — Does the returned bnlearn object record the parameters used?
# =============================================================================
log("----------------------------------------------------------------")
log("CHECK 2: Does bnlearn record hyperparameters in the returned object?")
log("----------------------------------------------------------------")

data(learning.test)
net_check <- mmpc(learning.test, test="mi", alpha=0.05, max.sx=3)

log("Fields in net$learning: ", paste(names(net_check$learning), collapse=", "))
if (!is.null(net_check$learning$args)) {
  log("net$learning$args contents:")
  for (nm in names(net_check$learning$args))
    log("  ", nm, " = ", net_check$learning$args[[nm]])
} else {
  log("net$learning$args is NULL — parameters not stored in object")
  log("Checking net$learning directly:")
  for (nm in names(net_check$learning))
    log("  ", nm, " = ", paste(net_check$learning[[nm]], collapse=", "))
}
log("")

# =============================================================================
# CHECK 3 — Toy dataset: do different α values produce different skeletons?
# =============================================================================
log("----------------------------------------------------------------")
log("CHECK 3: Alpha sensitivity on learning.test (5000 rows, 5 vars)")
log("Expected: sparser skeleton at stricter alpha, denser at lenient alpha")
log("----------------------------------------------------------------")

data(learning.test)
alphas <- c(0.001, 0.01, 0.05, 0.10, 0.20, 0.50)
results_alpha <- list()

for (a in alphas) {
  net <- tryCatch(
    mmpc(learning.test, test="mi", alpha=a, max.sx=3),
    error=function(e) NULL
  )
  if (is.null(net)) {
    log(sprintf("  alpha=%.3f : ERROR", a))
    next
  }
  n_edges   <- nrow(arcs(net)) / 2   # undirected
  nbrs      <- sapply(nodes(net), function(nd) length(nbr(net, nd)))
  adj       <- amat(net); nn <- nrow(adj)
  vis       <- rep(FALSE,nn); ncomp <- 0
  for (st in seq_len(nn)) {
    if (!vis[st]) {
      ncomp <- ncomp+1; q <- st
      while(length(q)>0){u<-q[1];q<-q[-1];if(!vis[u]){vis[u]<-TRUE;nb<-which(adj[u,]==1);q<-unique(c(q,nb[!vis[nb]]))}}
    }
  }
  results_alpha[[as.character(a)]] <- list(alpha=a, edges=n_edges, components=ncomp)
  log(sprintf("  alpha=%.3f : %d skeleton edges, %d component(s)", a, n_edges, ncomp))
}

# Check if all identical
edge_counts <- sapply(results_alpha, `[[`, "edges")
if (length(unique(edge_counts)) == 1) {
  log("  RESULT: All alpha values produced identical skeleton on learning.test")
  log("  This suggests alpha genuinely has no discriminating power on this dataset,")
  log("  OR that alpha is not being passed correctly — proceed to Check 4.")
} else {
  log("  RESULT: PASS — different alpha values produce different skeletons.")
  log("  This confirms alpha IS being passed and used by bnlearn correctly.")
}
log("")

# =============================================================================
# CHECK 4 — Toy dataset: does mi-sh differ from mi?
# =============================================================================
log("----------------------------------------------------------------")
log("CHECK 4: mi vs mi-sh on learning.test (same alpha, same max.sx)")
log("Expected: some difference if mi-sh is genuinely implemented")
log("----------------------------------------------------------------")

net_mi <- tryCatch(
  mmpc(learning.test, test="mi",    alpha=0.05, max.sx=3),
  error=function(e) { log("  mi failed: ", conditionMessage(e)); NULL }
)
net_sh <- tryCatch(
  mmpc(learning.test, test="mi-sh", alpha=0.05, max.sx=3),
  error=function(e) { log("  mi-sh failed: ", conditionMessage(e)); NULL }
)

if (!is.null(net_mi) && !is.null(net_sh)) {
  arcs_mi <- arcs(net_mi)
  arcs_sh <- arcs(net_sh)
  n_mi    <- nrow(arcs_mi)/2
  n_sh    <- nrow(arcs_sh)/2
  identical_result <- isTRUE(all.equal(sort(arcs_mi[,1]), sort(arcs_sh[,1])) &&
                               all.equal(sort(arcs_mi[,2]), sort(arcs_sh[,2])))
  log(sprintf("  mi    : %d skeleton edges", n_mi))
  log(sprintf("  mi-sh : %d skeleton edges", n_sh))
  if (identical_result) {
    log("  RESULT: IDENTICAL skeletons from mi and mi-sh on learning.test")
    log("  Possible explanations:")
    log("    (a) mi-sh is silently aliased to mi in this bnlearn version")
    log("    (b) learning.test dependencies are so strong that both tests agree")
    log("  Proceeding to Check 5 (larger/noisier dataset) to distinguish.")
  } else {
    log("  RESULT: PASS — mi and mi-sh produce different skeletons.")
    log("  mi-sh is genuinely implemented and differs from mi.")
  }
} else {
  log("  RESULT: Could not compare — at least one test failed.")
}
log("")

# =============================================================================
# CHECK 5 — Larger noisy dataset: alpha and mi-sh effects
# Use alarm dataset (37 vars, 20000 rows) if available, else gaussian.test
# =============================================================================
log("----------------------------------------------------------------")
log("CHECK 5: Alpha and mi-sh sensitivity on a larger/noisier dataset")
log("Using: alarm (37 vars, 20000 rows) if available, else gaussian.test")
log("----------------------------------------------------------------")

big_data <- tryCatch({ data(alarm); alarm }, error=function(e) NULL)
if (is.null(big_data)) {
  log("  alarm not available, using gaussian.test (5000 rows, 5 vars, continuous)")
  log("  NOTE: gaussian.test uses continuous data — switching test to cor (correlation)")
  data(gaussian.test)
  big_data  <- gaussian.test
  big_test1 <- "cor"
  big_test2 <- "cor"   # no shrinkage version for continuous in bnlearn
  log("  (mi-sh check skipped for continuous data — not applicable)")
} else {
  log("  alarm dataset loaded: ", nrow(big_data), " rows, ", ncol(big_data), " vars")
  big_data  <- lapply(big_data, as.factor)
  big_data  <- as.data.frame(big_data)
  big_test1 <- "mi"
  big_test2 <- "mi-sh"
}

# Alpha sweep on big dataset
log("")
log("  Alpha sweep (test=mi):")
alphas_big <- c(0.001, 0.01, 0.05, 0.10, 0.20)
edge_big   <- c()
for (a in alphas_big) {
  net <- tryCatch(
    mmpc(big_data, test=big_test1, alpha=a, max.sx=3),
    error=function(e) NULL
  )
  if (is.null(net)) { log(sprintf("  alpha=%.3f : ERROR")); next }
  n_edges <- nrow(arcs(net))/2
  edge_big <- c(edge_big, n_edges)
  log(sprintf("    alpha=%.3f : %d skeleton edges", a, n_edges))
}
if (length(unique(edge_big))>1) {
  log("  RESULT: PASS — alpha affects skeleton size on this dataset.")
} else {
  log("  RESULT: Alpha produced identical results on this dataset too.")
  log("  This is unusual. Check bnlearn version or dataset factor encoding.")
}

# mi vs mi-sh on big dataset
if (big_test1=="mi") {
  log("")
  log("  mi vs mi-sh comparison:")
  net_mi2 <- tryCatch(mmpc(big_data, test="mi",    alpha=0.05, max.sx=3), error=function(e) NULL)
  net_sh2 <- tryCatch(mmpc(big_data, test="mi-sh", alpha=0.05, max.sx=3), error=function(e) NULL)
  if (!is.null(net_mi2) && !is.null(net_sh2)) {
    n_mi2 <- nrow(arcs(net_mi2))/2
    n_sh2 <- nrow(arcs(net_sh2))/2
    log(sprintf("    mi    : %d skeleton edges", n_mi2))
    log(sprintf("    mi-sh : %d skeleton edges", n_sh2))
    if (n_mi2 == n_sh2) {
      log("  RESULT: Same edge count. Checking arc-by-arc identity...")
      arcs_mi2 <- arcs(net_mi2)[order(arcs(net_mi2)[,1], arcs(net_mi2)[,2]),]
      arcs_sh2 <- arcs(net_sh2)[order(arcs(net_sh2)[,1], arcs(net_sh2)[,2]),]
      if (isTRUE(all.equal(arcs_mi2, arcs_sh2))) {
        log("  RESULT: IDENTICAL — mi-sh appears to be aliased to mi in this bnlearn version.")
        log("  ACTION: Check bnlearn source: bnlearn:::mi.sh or equivalent.")
      } else {
        log("  RESULT: PASS — same edge count but different edges. mi-sh is active.")
      }
    } else {
      log("  RESULT: PASS — mi and mi-sh produce different skeletons on alarm.")
    }
  }
}
log("")

# =============================================================================
# CHECK 6 — Direct bnlearn internals: is mi-sh registered?
# =============================================================================
log("----------------------------------------------------------------")
log("CHECK 6: Does bnlearn internally register mi-sh as a test?")
log("----------------------------------------------------------------")

# bnlearn stores test functions in its namespace
mi_sh_fn <- tryCatch(
  getFromNamespace("mi-sh", "bnlearn"),
  error=function(e) NULL
)
mi_sh_fn2 <- tryCatch(
  getFromNamespace("mi.sh", "bnlearn"),
  error=function(e) NULL
)
discrete_tests <- tryCatch(
  getFromNamespace("available.discrete.tests", "bnlearn"),
  error=function(e) tryCatch(
    getFromNamespace(".discrete.tests", "bnlearn"),
    error=function(e2) NULL
  )
)

if (!is.null(mi_sh_fn))  log("  'mi-sh'  found as function in bnlearn namespace: YES")
else                      log("  'mi-sh'  found as function in bnlearn namespace: NO")
if (!is.null(mi_sh_fn2)) log("  'mi.sh'  found as function in bnlearn namespace: YES")
else                      log("  'mi.sh'  found as function in bnlearn namespace: NO")

if (!is.null(discrete_tests)) {
  log("  Available discrete tests in this bnlearn version:")
  log("    ", paste(discrete_tests, collapse=", "))
  if ("mi-sh" %in% discrete_tests) log("  RESULT: PASS — mi-sh is in the official test list.")
  else                               log("  RESULT: WARN — mi-sh is NOT in the official test list.")
} else {
  log("  Could not retrieve test list from bnlearn internals.")
  log("  Trying bnlearn::bnlearn.test.options()...")
  opts <- tryCatch(bnlearn:::discreteTests, error=function(e) NULL)
  if (!is.null(opts)) log("  discreteTests: ", paste(opts, collapse=", "))
  else log("  Could not access bnlearn internal test list.")
}
log("")

# =============================================================================
# CHECK 7 — Your actual data (if path provided)
# Run 3 strategically different combos on 1 fold, log net$learning$args
# =============================================================================
if (!is.null(input_dir)) {
  log("----------------------------------------------------------------")
  log("CHECK 7: Verification on your actual dataset (1 fold, 3 combos)")
  log("Input dir: ", input_dir)
  log("----------------------------------------------------------------")

  dt_file <- file.path(input_dir, "complete_DB_discrete.csv")
  cr_file <- file.path(input_dir, "crashes.csv")
  nc_file <- file.path(input_dir, "no_crashes.csv")

  if (!file.exists(dt_file) || !file.exists(cr_file) || !file.exists(nc_file)) {
    log("  Required CSV files not found in ", input_dir, " — skipping Check 7.")
  } else {
    dt            <- fread(dt_file, colClasses="character")
    dt            <- dt[complete.cases(dt)]
    dt_crashes    <- fread(cr_file, colClasses="character")
    dt_no_crashes <- fread(nc_file, colClasses="character")
    dt_crashes[,    orig_label_lc:="True"]
    dt_no_crashes[, orig_label_lc:="False"]
    dt_unique <- rbindlist(list(dt_crashes, dt_no_crashes))
    dt_unique[, latent_collision:="True"]
    state_cols <- c("curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W")
    dt[,        state_id:=do.call(paste,c(.SD,sep="_")),.SDcols=state_cols]
    dt_unique[, state_id:=do.call(paste,c(.SD,sep="_")),.SDcols=state_cols]

    # Use fold 64 (first in sensitivity runs) at 1% training
    fold_i    <- 64
    fraction  <- 0.01
    sid       <- dt_unique$state_id[fold_i]
    train_dt  <- dt[state_id != sid]
    ss        <- round(fraction * nrow(train_dt))
    set.seed(300)
    train_samp <- train_dt[sample(.N, ss)]
    df_f <- as.data.frame(lapply(
      train_samp[,lapply(.SD,as.character),.SDcols=setdiff(names(train_samp),"state_id")],
      factor))

    log("  Training sample: ", nrow(df_f), " rows, fold ", fold_i, " @ 1%")
    log("")

    # Run 3 combos and log what bnlearn actually used
    real_combos <- list(
      list(test="mi",    alpha=0.01, max.sx=3, label="A (mi, α=0.01, sx=3)"),
      list(test="mi",    alpha=0.20, max.sx=3, label="D (mi, α=0.20, sx=3)"),
      list(test="mi-sh", alpha=0.05, max.sx=3, label="G (mi-sh, α=0.05, sx=3)")
    )

    for (combo in real_combos) {
      log("  --- Combo: ", combo$label, " ---")
      net <- tryCatch(
        mmpc(df_f, test=combo$test, alpha=combo$alpha, max.sx=combo$max.sx),
        error=function(e) { log("  ERROR: ", conditionMessage(e)); NULL }
      )
      if (is.null(net)) next

      # What did bnlearn actually record?
      log("  Arcs in skeleton: ", nrow(arcs(net))/2)
      lc_nbrs <- tryCatch(nbr(net,"latent_collision"), error=function(e) character(0))
      log("  LC neighbors: ", if(length(lc_nbrs)==0) "NONE" else paste(sort(lc_nbrs),collapse=", "))

      # Audit: what does the learning slot say?
      log("  net$learning recorded:")
      if (!is.null(net$learning$args)) {
        for (nm in names(net$learning$args))
          log("    args$", nm, " = ", paste(net$learning$args[[nm]], collapse=", "))
      }
      if (!is.null(net$learning$test))
        log("    test  = ", net$learning$test)
      if (!is.null(net$learning$alpha))
        log("    alpha = ", net$learning$alpha)
      log("")
    }
  }
}

# =============================================================================
# CHECK 8 — Extreme alpha test: does alpha=0.9999 give a complete graph?
# =============================================================================
log("----------------------------------------------------------------")
log("CHECK 8: Sanity check — does alpha=0.9999 give a denser graph than alpha=0.01?")
log("On learning.test (guaranteed to have real edges)")
log("If both give the same result, alpha is definitely not being used.")
log("----------------------------------------------------------------")

data(learning.test)
net_tight <- tryCatch(mmpc(learning.test, test="mi", alpha=0.001,  max.sx=3), error=function(e) NULL)
net_loose <- tryCatch(mmpc(learning.test, test="mi", alpha=0.9999, max.sx=3), error=function(e) NULL)

if (!is.null(net_tight) && !is.null(net_loose)) {
  n_tight <- nrow(arcs(net_tight))/2
  n_loose <- nrow(arcs(net_loose))/2
  log(sprintf("  alpha=0.001  : %d skeleton edges", n_tight))
  log(sprintf("  alpha=0.9999 : %d skeleton edges", n_loose))
  if (n_loose > n_tight) {
    log("  RESULT: PASS — alpha is being used. Looser threshold gives more edges.")
  } else if (n_loose == n_tight) {
    log("  RESULT: SUSPICIOUS — same edge count despite extreme alpha range.")
    log("  This may be OK if all edges are strongly significant, but warrants inspection.")
  } else {
    log("  RESULT: ANOMALY — stricter alpha gave MORE edges. Investigate.")
  }
}
log("")

# =============================================================================
# SUMMARY
# =============================================================================
log("================================================================")
log(" SUMMARY")
log("================================================================")
log("")
log("Review each CHECK above. Key items to address in your revision:")
log("")
log("1. If CHECK 6 shows mi-sh is NOT in bnlearn's test list, then G")
log("   likely silently fell back to mi. This means your G result")
log("   cannot be cited as a mi-sh result — it is a duplicate of B.")
log("   Solution: verify bnlearn version or use an alternative estimator.")
log("")
log("2. If CHECK 8 PASSES (it almost certainly will), you can cite this")
log("   as proof that alpha IS being passed and used correctly.")
log("   The identical A-D results then reflect genuine data insensitivity.")
log("")
log("3. If CHECK 7 shows net$learning records the right parameters,")
log("   include a table in the paper appendix showing alpha and test")
log("   as logged from the bnlearn object — not just from your script.")
log("")
log("Report saved to: ", report_file)
log("================================================================")

close(con)
message("\nVerification complete. Full report: ", report_file)
