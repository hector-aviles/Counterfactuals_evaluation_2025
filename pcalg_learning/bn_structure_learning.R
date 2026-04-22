#!/usr/bin/env Rscript
# =============================================================================
# bn_structure_learning.R
#
# Factored bridge layer between structure learning algorithms (pcalg, bnlearn)
# and the downstream bn.fit + .pl translation pipeline.
#
# Design goals:
#   - Swap algorithms by changing one argument to `learn_structure()`
#   - Handles DAG / CPDAG detection and extension transparently
#   - Returns a consistent result object regardless of algorithm used
#   - Logs structure type per fold for reproducibility
# =============================================================================


# -----------------------------------------------------------------------------
# 1. ALGORITHM RUNNERS
#    One function per algorithm. Each accepts (data, ...) and returns a
#    raw result list: list(amat = <p×p integer matrix>, algo = "<name>")
#    where amat follows bnlearn convention: amat[i,j]=1 means i → j
# -----------------------------------------------------------------------------

#' Run bnlearn hill-climbing (hc)
#' @param data  data.frame of factors
#' @param ...   passed to bnlearn::hc()
run_hc <- function(data, ...) {
  net <- bnlearn::hc(data, ...)
  list(
    amat = bnlearn::amat(net),
    algo = "hc",
    raw  = net
  )
}

#' Run bnlearn mmpc (returns skeleton / CPDAG)
#' @param data  data.frame of factors
#' @param ...   passed to bnlearn::mmpc()
run_mmpc <- function(data, ...) {
  net <- bnlearn::mmpc(data, ...)
  list(
    amat = bnlearn::amat(net),
    algo = "mmpc",
    raw  = net
  )
}

#' Run pcalg GES
#' @param data  data.frame of factors
#' @param ...   passed to pcalg::ges()
run_ges <- function(data, ...) {
  score <- new("DiscreteL0Pen", data = data,
               lambda = 0.5 * log(nrow(data)))
  res   <- pcalg::ges(score, ...)
  amat  <- as(res$essgraph, "matrix") * 1L  # logical → integer
  # pcalg convention: amat[i,j]=1 means i → j (same as bnlearn), verify below
  list(
    amat = amat,
    algo = "ges",
    raw  = res
  )
}

#' Run pcalg PC algorithm (discrete data via G² test)
#' @param data  data.frame of factors
#' @param alpha significance level for CI tests (default 0.05)
#' @param ...   passed to pcalg::pc()
run_pc <- function(data, alpha = 0.05, ...) {
  n     <- nrow(data)
  p     <- ncol(data)
  nlev  <- sapply(data, nlevels)
  suf   <- list(dm = data, nlev = nlev, adaptDF = FALSE)
  res   <- pcalg::pc(
    suffStat  = suf,
    indepTest = pcalg::disCItest,
    alpha     = alpha,
    labels    = colnames(data),
    skel.method = "stable",
    ...
  )
  amat  <- as(res, "amat")  # integer adjacency matrix
  list(
    amat = amat,
    algo = "pc",
    raw  = res
  )
}

#' Run pcalg GIES (requires intervention targets — set to observational by default)
#' @param data  data.frame of factors
#' @param targets list of intervention targets; default = observational only
#' @param ...   passed to pcalg::gies()
run_gies <- function(data, targets = list(integer(0)), ...) {
  score <- new("DiscreteL0Pen", data = data,
               lambda = 0.5 * log(nrow(data)),
               targets = targets)
  res   <- pcalg::gies(score, targets = targets, ...)
  amat  <- as(res$essgraph, "matrix") * 1L
  list(
    amat = amat,
    algo = "gies",
    raw  = res
  )
}


# -----------------------------------------------------------------------------
# 2. DISPATCH TABLE
#    Maps algorithm name string → runner function.
#    Add a new algorithm here without touching learn_structure().
# -----------------------------------------------------------------------------

.algo_registry <- list(
  hc   = run_hc,
  mmpc = run_mmpc,
  ges  = run_ges,
  pc   = run_pc,
  gies = run_gies
)


# -----------------------------------------------------------------------------
# 3. GRAPH TYPE DETECTION
#    Returns "dag", "cpdag", or "pdag" (partially oriented).
# -----------------------------------------------------------------------------

#' Classify an adjacency matrix as DAG, CPDAG/PDAG, or unknown
#' @param amat integer p×p adjacency matrix (bnlearn convention)
#' @return character: "dag" | "cpdag" | "invalid"
classify_graph <- function(amat) {
  # Undirected edge: both amat[i,j]==1 and amat[j,i]==1
  has_undirected <- any(amat == 1L & t(amat) == 1L)

  # Cycle check via topological sort (simple DFS)
  has_cycle <- function(m) {
    p <- nrow(m)
    visited <- integer(p)   # 0=unvisited, 1=in-stack, 2=done
    cyclic  <- FALSE
    dfs <- function(v) {
      if (cyclic) return()
      visited[v] <<- 1L
      nbrs <- which(m[v, ] == 1L & m[, v] == 0L)  # directed out-edges only
      for (w in nbrs) {
        if (visited[w] == 1L) { cyclic <<- TRUE; return() }
        if (visited[w] == 0L) dfs(w)
      }
      visited[v] <<- 2L
    }
    for (v in seq_len(p)) if (visited[v] == 0L) dfs(v)
    cyclic
  }

  if (!has_undirected && !has_cycle(amat)) return("dag")
  if (has_undirected)                      return("cpdag")
  return("invalid")
}


# -----------------------------------------------------------------------------
# 4. GRAPH CONVERSION
#    amat → bnlearn bn object, with CPDAG → DAG extension if needed.
# -----------------------------------------------------------------------------

#' Convert adjacency matrix to bnlearn bn, extending CPDAG → DAG if necessary
#'
#' @param amat        integer p×p adjacency matrix
#' @param node_names  character vector of node names
#' @param graph_type  output of classify_graph(); computed if NULL
#' @param strict      passed to bnlearn::cextend(); FALSE = warn instead of error
#'
#' @return list(
#'   bn           = bnlearn bn object (always a DAG),
#'   graph_type   = "dag" | "cpdag",
#'   was_extended = logical
#' )
amat_to_bn_dag <- function(amat, node_names, graph_type = NULL, strict = FALSE) {
  if (is.null(graph_type)) graph_type <- classify_graph(amat)

  if (graph_type == "invalid") {
    stop("amat_to_bn_dag: adjacency matrix is neither a DAG nor a CPDAG.")
  }

  # Build bnlearn object from amat
  g <- bnlearn::empty.graph(node_names)
  bnlearn::amat(g) <- amat

  was_extended <- FALSE

  if (graph_type == "cpdag") {
    # Try to extend CPDAG to a consistent DAG extension
    g <- tryCatch(
      bnlearn::cextend(g, strict = strict),
      error = function(e) {
        stop("cextend() failed: ", conditionMessage(e),
             "\nConsider using pdag2dag() from pcalg as fallback.")
      }
    )
    was_extended <- TRUE
  }

  # Final acyclicity check
  if (!bnlearn::acyclic(g)) {
    stop("amat_to_bn_dag: resulting graph is not acyclic after extension.")
  }

  list(
    bn           = g,
    graph_type   = graph_type,
    was_extended = was_extended
  )
}


# -----------------------------------------------------------------------------
# 5. TOP-LEVEL ENTRY POINT
#    This is the only function the LOOCV loop needs to call.
# -----------------------------------------------------------------------------

#' Learn BN structure from data and return a fitted bn + metadata
#'
#' Drop-in replacement for the hc() + bn.fit() block in the LOOCV loop.
#'
#' @param data      data.frame of factors (training sample)
#' @param algorithm character: one of names(.algo_registry) — "hc","ges","pc","mmpc","gies"
#' @param algo_args named list of extra arguments forwarded to the runner
#' @param fit_method passed to bnlearn::bn.fit() (default "mle")
#'
#' @return list(
#'   bn_fit       = fitted bn object ready for .pl translation,
#'   bn_struct    = unfitted bn structure,
#'   graph_type   = "dag" | "cpdag",
#'   was_extended = logical,
#'   algorithm    = character
#' )
learn_structure <- function(data,
                            algorithm = "hc",
                            algo_args = list(),
                            fit_method = "mle") {

  # 1. Dispatch to runner
  runner <- .algo_registry[[algorithm]]
  if (is.null(runner)) {
    stop("Unknown algorithm '", algorithm, "'. ",
         "Available: ", paste(names(.algo_registry), collapse = ", "))
  }
  raw_result <- do.call(runner, c(list(data = data), algo_args))

  # 2. Classify graph
  graph_type <- classify_graph(raw_result$amat)
  message("  [learn_structure] algo=", algorithm,
          "  graph_type=", graph_type)

  # 3. Convert to bnlearn DAG
  conv <- amat_to_bn_dag(
    amat       = raw_result$amat,
    node_names = colnames(data),
    graph_type = graph_type
  )

  # 4. Parameter learning
  bn_fit <- bnlearn::bn.fit(
    conv$bn,
    data    = data,
    method  = fit_method,
    replace.unidentifiable = TRUE
  )

  list(
    bn_fit       = bn_fit,
    bn_struct    = conv$bn,
    graph_type   = graph_type,
    was_extended = conv$was_extended,
    algorithm    = algorithm
  )
}
