#!/usr/bin/env Rscript
# =============================================================================
# Excerpt: how to integrate bn_structure_learning.R into CBNs_LOOCV_training.R
#
# Replace lines ~268-295 (the hc + bn.fit block) with the block below.
# Source the bridge file once at the top of your script.
# =============================================================================

# --- At the top of CBNs_LOOCV_training.R, add: ---
source("bn_structure_learning.R")

# Pick algorithm here — only this line changes when you want to swap:
BN_ALGORITHM  <- "ges"          # "hc" | "ges" | "pc" | "mmpc" | "gies"
BN_ALGO_ARGS  <- list()         # e.g. list(alpha = 0.01) for "pc"
BN_FIT_METHOD <- "mle"


# =============================================================================
# REPLACEMENT BLOCK — inside the LOOCV fold loop (replaces ~lines 268-295)
# =============================================================================

      # --- Structure + parameter learning ---
      start_time_struct <- Sys.time()

      struct_result <- tryCatch(
        learn_structure(
          data      = df_factor,
          algorithm = BN_ALGORITHM,
          algo_args = BN_ALGO_ARGS,
          fit_method = BN_FIT_METHOD
        ),
        error = function(e) {
          message("  [fold ", i, "] learn_structure() failed: ", conditionMessage(e))
          NULL
        }
      )

      end_time_struct <- Sys.time()
      elapsed_time_struct <- as.numeric(end_time_struct - start_time_struct, units = "secs")

      # Skip fold if structure learning failed (e.g. CPDAG could not be extended)
      if (is.null(struct_result)) {
        message("  Skipping fold ", i, " — could not produce a valid DAG.")
        next
      }

      # Unpack — same names as before so rest of loop is unchanged
      network_structure <- struct_result$bn_struct
      bn_fit            <- struct_result$bn_fit

      # --- Log structure type to numeralia (new field) ---
      numeralia_line <- sprintf(
        "Fold %d: algo=%s graph_type=%s extended=%s Training Time = %.2f seconds, Samples Removed = %d, Train Sample Size = %d",
        i,
        struct_result$algorithm,
        struct_result$graph_type,
        struct_result$was_extended,
        elapsed_time_struct,   # structural time only; add param time if needed
        num_removed,
        train_sample_size
      )

      # --- Save .dot (write.dot expects a bn, works for both DAG and CPDAG) ---
      output_base <- file.path(cbns_dir, paste0("cBN_", i))
      model_dot   <- paste0(output_base, ".dot")
      bnlearn::write.dot(network_structure, file = model_dot)

      # --- Save .net ---
      model_net_fn <- paste0(output_base, ".net")
      try(bnlearn::write.net(model_net_fn, bn_fit), silent = TRUE)

      # --- .pl translation continues unchanged from here ---
      # (the bn_fit object has the same structure as before)
