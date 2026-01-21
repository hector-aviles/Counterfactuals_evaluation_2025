#!/usr/bin/env Rscript

suppressMessages({
  if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table", repos = "https://cloud.r-project.org")
  if (!requireNamespace("conflicted", quietly = TRUE)) install.packages("conflicted", repos = "https://cloud.r-project.org")
  if (!requireNamespace("bnlearn", quietly = TRUE)) install.packages("bnlearn", repos = "https://cloud.r-project.org")
  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", repos = "https://cloud.r-project.org")
  if (!requireNamespace("stringr", quietly = TRUE)) install.packages("stringr", repos = "https://cloud.r-project.org")
  if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr", repos = "https://cloud.r-project.org")
  if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse", repos = "https://cloud.r-project.org")
  if (!requireNamespace("arules", quietly = TRUE)) install.packages("arules", repos = "https://cloud.r-project.org")
})

library(data.table)
library(conflicted)
conflict_prefer("setdiff", "base")
library(bnlearn)
library(dplyr)
library(stringr)
library(tidyr)
library(tidyverse)
library(arules)

percentage <- 100
fraction <- percentage / 100
# Set the decimal rounding
dec_round <- 7

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript script.R <shared_csv_path> [reps]")
input_subdir1 <- args[1]
reps_arg <- ifelse(length(args) >= 2, as.numeric(args[2]), 5)
reps <- 1:reps_arg
# -----------------------
# Parse percentages argument (optional third CLI argument)
# -----------------------
if (length(args) >= 3) {
  # Expecting a comma-separated string, e.g., "01,25,50,75,90" or "10,20,30"
  percentages_str <- args[3]
  percentages <- strsplit(percentages_str, ",")[[1]]
  percentages <- trimws(percentages)  # remove any accidental whitespace
  message("Percentages provided via CLI: ", paste(percentages, collapse = ", "))
} else {
  # default percentages if not provided
  percentages <- c("01", "25", "50", "75", "90")
  message("No percentages argument provided; using default: ", paste(percentages, collapse = ", "))
}

message("Input directory: ", input_subdir1)
message("Number of repetitions: ", reps)


# -----------------------
# Configuration
# -----------------------
seeds <- c(300, 456, 211, 26, 500, 1001, 724, 881, 91, 255)   # one seed per rep (same as original)
shared_csv_path <- "./Shared_CSVs"   # location where complete_DB_discrete.csv, crashes.csv, no_crashes.csv live

# Use current working directory as base
base_wd <- getwd()
message("Working directory: ", base_wd)

# -----------------------
# Utility functions
# -----------------------
adjust_values <- function(freq) {
   freq[is.na(freq)] <- 0.0
   freq[ freq <= 0.0001 ] <- 0.001
   total_sum <- sum(freq)
   if (total_sum == 0.0) {
      freq <- rep(1/length(freq), length(freq))
   } else {
      if (total_sum > 1) freq <- freq / total_sum
   }
   return(freq)
}

report_progress <- function(step_msg, progress_file = NULL) {
  message(step_msg)
  if (!is.null(progress_file)) cat(step_msg, "\n", file = progress_file, append = TRUE)
}

# -----------------------
# Input validation: shared CSVs exist
# -----------------------
input_files <- file.path(shared_csv_path, c("complete_DB_discrete.csv", "crashes.csv", "no_crashes.csv"))
for (f in input_files) {
  if (!file.exists(f)) stop("Input file missing: ", f)
}

# -----------------------
# Read core datasets once (we'll copy per rep to ensure reproducibility)
# -----------------------
# --- Read datasets ---
dt <- fread(file.path(input_subdir1, "complete_DB_discrete.csv"), colClasses = "character")
dt <- dt[complete.cases(dt)]

dt_crashes <- fread(file.path(input_subdir1, "crashes.csv"), colClasses = "character")
dt_crashes[, orig_label_lc := "True"]
dt_no_crashes <- fread(file.path(input_subdir1, "no_crashes.csv"), colClasses = "character")
dt_no_crashes[, orig_label_lc := "False"]

# Combine crashes and no-crashes into unique examples (LOOCV set)
dt_unique <- rbindlist(list(dt_crashes, dt_no_crashes))
dt_unique[, latent_collision := "True"]

# Add state_id for filtering
state_cols <- c("curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W")
dt[, state_id := do.call(paste, c(.SD, sep = "_")), .SDcols = state_cols]
dt_unique[, state_id := do.call(paste, c(.SD, sep = "_")), .SDcols = state_cols]

for (nm in c("dt", "dt_unique")) {
  dt_check <- get(nm)
  if (anyNA(dt_check$state_id)) stop("NA values found in state_id in ", nm)
}

# --- Initialize global summary ---
global_summary <- data.table(
  Repetition = integer(),
  Fold = integer(),
  TrainingTime_s = numeric(),
  SamplesRemoved = integer(),
  TrainSampleSize = integer()
)

# -----------------------
# Main loops: repetitions and percentages
# -----------------------
for (r_idx in seq_along(reps)) {
  rep_num <- reps[r_idx]
  seed <- seeds[r_idx]
  set.seed(seed)

  message("\n==============================")
  message("Starting repetition: ", rep_num)
  message("==============================\n")

  # Create rep directory and test_data subdir (if not exists)
  rep_dir <- file.path(base_wd, sprintf("rep_%d", rep_num))
  if (!dir.exists(rep_dir)) dir.create(rep_dir, recursive = TRUE)
  rep_test_dir <- file.path(rep_dir, "test_data")
  if (!dir.exists(rep_test_dir)) dir.create(rep_test_dir, recursive = TRUE)

  # Copy master dt and dt_unique to local variables for this repetition
  dt <- copy(dt)
  dt_crashes <- copy(dt_crashes)
  dt_no_crashes <- copy(dt_no_crashes)
  dt_unique <- copy(dt_unique)

  # Number of LOOCV folds = number of rows in dt_unique
  n_folds <- nrow(dt_unique)
  message("LOOCV folds (unique examples): ", n_folds)

  # We'll generate test data once per repetition (as you previously did).
  # To mirror prior behavior, we'll generate test datasets during the first percentage "01"
  # but we'll still loop percentages afterwards. If you prefer a different rule, tell me.
  for (percentage in percentages) {
    message("\n--- Repetition ", rep_num, " | Percentage ", percentage, " ---\n")

    # Each percentage has its own training_data and cBNs under the rep folder
    pct_dir <- file.path(rep_dir, percentage)
    training_dir <- file.path(pct_dir, "training_data")
    cbns_dir <- file.path(pct_dir, "cBNs")
    for (d in c(training_dir, cbns_dir)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

    # Sampling fraction
    fraction <- as.numeric(percentage) / 100
    if (fraction <= 0 | fraction > 1) stop("Percentage must be between 0 and 100")

    # Initialize numeralia lists for this percentage (kept across folds)
    training_times <- numeric()
    samples_removed <- numeric()
    train_sample_sizes <- numeric()

    # LOOCV loop over unique examples
    for (i in 1:n_folds) {
      message("Processing fold ", i, " of ", n_folds)

      # --- Create test dataset for the i-th unique example (only once per repetition in previous flow)
      current_test_example <- dt_unique[i, ]
      report_progress("Creating test dataset...")

      action_list <- c("change_to_left", "change_to_right", "cruise", "keep", "swerve_left", "swerve_right")
      # duplicate row for all six actions
      dt_test <- current_test_example[rep(1:.N, each = 6)][, iaction := rep(action_list, times = nrow(current_test_example))]
      message("Done. dt_test dims: ", paste(dim(dt_test), collapse = " x "))

      if (anyNA(dt_test$state_id)) stop("NA values found in state_id in dt_test")

      # Save test file into rep_<r>/test_data/ (one file per fold)
      test_file <- file.path(rep_test_dir, paste0("test_fold_", i, ".csv"))
      dt_test_to_save <- dt_test[, lapply(.SD, as.character), .SDcols = setdiff(names(dt_test), "state_id")]
      fwrite(dt_test_to_save, test_file, quote = TRUE)
      report_progress(paste("Test data saved to:", test_file))

      # --- Build training dataset: remove all rows matching current_state_id from dt (master dataset)
      current_state_id <- dt_unique$state_id[i]
      num_removed <- sum(dt$state_id == current_state_id)
      samples_removed <- c(samples_removed, num_removed)
      train_dt <- dt[state_id != current_state_id]   # all remaining rows

      # Validate no data leakage: check the state_id absent in training
      if (current_state_id %in% train_dt$state_id) {
        stop("Data leakage detected: test state_id found in training set for fold ", i)
      }

      # --- Sample training set with at least one per action if possible
      action_list_unique <- unique(train_dt$action)
      num_actions <- length(action_list_unique)
      sample_size <- round(fraction * nrow(train_dt))
      if (sample_size < num_actions) {
        message("Warning: sample_size (", sample_size, ") smaller than number of actions (", num_actions, ") in fold ", i)
        train_sample <- train_dt[sample(.N, min(sample_size, .N))]
      } else {
        min_samples <- train_dt[, .SD[sample(.N, min(1, .N))], by = action]
        remaining_size <- sample_size - nrow(min_samples)
        if (remaining_size > 0) {
          remaining_sample <- train_dt[sample(.N, remaining_size)]
          train_sample <- rbindlist(list(min_samples, remaining_sample))
        } else {
          train_sample <- min_samples
        }
      }

      # Record sample size
      train_sample_size <- nrow(train_sample)
      train_sample_sizes <- c(train_sample_sizes, train_sample_size)
      message("Train sample size: ", train_sample_size)

      # Show action distribution
      action_dist <- train_sample[, .N, by = action]
      message("Action distribution for fold ", i, ":\n", paste(capture.output(print(action_dist)), collapse = "\n"))

      # Save training sample to rep_<r>/<percentage>/training_data/
      train_sample_to_save <- train_sample[, lapply(.SD, as.character), .SDcols = setdiff(names(train_sample), "state_id")]
      training_file <- file.path(training_dir, paste0("training_data_", percentage, "_fold_", i, ".csv"))
      #if (exists("train_sample") && nrow(train_sample) > 0) {
      #  fwrite(train_sample_to_save, training_file)
      #  report_progress(paste("Training data saved to:", training_file))
      #} else {
      #  warning("Training data not saved: train_sample is missing or empty for fold ", i)
      #}

      # --- Prepare for training: convert to factors
      df_factor <- as.data.frame(lapply(train_sample_to_save, factor))

      # --- Structural learning (hc) timing
      start_time_struct <- Sys.time()
      
      # MMPC learns a local PC set for each node, from which a skeleton is derived
      net_mmpc <- mmpc(
        df_factor,
        test = "mi",        # mutual information test (good for discrete vars)
        alpha = 0.01,       # significance level for CI tests
        max.sx = 2       # max conditioning set size
      )
      
      # Extract arc constraints from MMPC
      wl <- arcs(net_mmpc)   # edges allowed (skeleton)
      
      # Extract MMPC arcs
      wl_raw <- arcs(net_mmpc)

      # Enforce unique undirected edges
      wl_unique <- t(apply(wl_raw, 1, function(x) sort(x)))
      wl_unique <- unique(wl_unique)
      
      network_structure <- hc(df_factor, whitelist = wl_unique,  
         score = "bic", debug = FALSE, restart = 0, perturb = 1, 
         max.iter = Inf, maxp = Inf,  optimized = TRUE)

      end_time_struct <- Sys.time()
      elapsed_time_struct <- as.numeric(end_time_struct - start_time_struct, units = "secs")

      # --- Save network plot and dot
      output_base <- file.path(cbns_dir, paste0("cBN_", i))
      model_dot <- paste0(output_base, ".dot")
      write.dot(network_structure, file = model_dot)
      # Plot to PNG
      #model_png <- paste0(output_base, ".png")
      #try({
      #  png(model_png)
      #  plot(network_structure)
      #  dev.off()
      #}, silent = TRUE)
      # Also create PS via dot if dot is available on system
      try({
        cmd <- paste("dot -Tps", shQuote(model_dot), "-o", shQuote(paste0(output_base, ".ps")))
        system(cmd, intern = TRUE)
      }, silent = TRUE)

      #print(network_structure)

      # --- Parametric learning (bn.fit)
      start_time_param <- Sys.time()
      bn_fit <- bn.fit(network_structure, data = df_factor, method = "mle", replace.unidentifiable = TRUE)
      end_time_param <- Sys.time()
      elapsed_time_param <- as.numeric(end_time_param - start_time_param, units = "secs")

      total_training_time <- elapsed_time_struct + elapsed_time_param
      training_times <- c(training_times, total_training_time)

      # --- Write numeralia (append)
      numeralia_file <- file.path(cbns_dir, "training_numeralia.txt")
      numeralia_line <- sprintf("Fold %d: Training Time = %.2f seconds, Samples Removed = %d, Train Sample Size = %d", i, total_training_time, num_removed, train_sample_size)
      if (!file.exists(numeralia_file)) {
        writeLines(numeralia_line, numeralia_file)
      } else {
        write(sprintf("\n%s", numeralia_line), file = numeralia_file, append = TRUE)
      }

      # --- Write parameters to .net file (bnlearn write.net)
      model_net_fn <- paste0(output_base, ".net")
      try({
        write.net(model_net_fn, bn_fit)
      }, silent = TRUE)

      # --- Write the .pl probabilistic facts and rules (original logic)
      # Write the probabilistic logic program
      output_pl <- paste0(output_base, ".pl")
      message("Writing probabilistic logic program to: ", output_pl)
      output_file <- file(output_pl, "w")
      writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%", con = output_file)
      writeLines("% Exogenous variables", con = output_file)
      writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n", con = output_file)

      # Extract parent variables and insert (unobserved) exogenous variables
      u_idx <- 0
      for (rv in bn_fit) {
        rv_name <- rv$node
        rv_values <- dimnames(rv$prob)[[1]]
        rv_numval = length(rv_values)
      
        if (identical(rv$parents, character(0))){
          #message("Processing RV: ", rv_name)
          #flush.console()
          df2 <- as.data.frame(rv$prob)
          df2[] <- lapply(df2, as.character)
          df2$Freq <- as.numeric(df2$Freq)
      
          # apply adjust_values if you want to avoid tiny probs (keeps original behavior)
          freq <- adjust_values(df2$Freq)
          df2$Freq <- freq
      
          if (rv_numval == 2 && identical(tolower(rv_values), c("false","true"))) {
              u_idx <- u_idx + 1
              fact <- paste0(format(round(df2[2,]$Freq, dec_round), nsmall=dec_round, scientific=FALSE), "::u", u_idx, ".\n", rv_name, " :- u", u_idx, ".\n")
              writeLines(fact, con = output_file)
            } else {
              # Non-Boolean variables (multi-valued or binary non-Boolean)
              probs <- sapply(1:rv_numval, function(i) format(round(df2[i,]$Freq, dec_round), nsmall=dec_round, scientific=FALSE))
              values <- df2[,1]
      
              # Construct u_variable_name prefix
              u_name <- paste0("u_", rv_name)
      
              ad_parts <- mapply(function(p, v) paste0(p, "::", u_name, "(", v, ")"), probs, values, SIMPLIFY = TRUE)
              ad_line <- paste(ad_parts, collapse="; ")
      
              writeLines(paste0(ad_line, "."), con = output_file)
              writeLines(paste0(rv_name, "(V) :- ", u_name, "(V).\n"), con = output_file)
           }
        }
      }
      
      # Extract rules (endogenous)
      for (rv in bn_fit) {
        rv_name <- rv$node
        rv_values <- dimnames(rv$prob)[[1]]
        rv_numval = length(rv_values)
      
        if (!identical(rv$parents, character(0))){
            #message("Processing RV: ", rv_name)
            #flush.console()
            df2 <- as.data.frame(rv$prob)
            df2[] <- lapply(df2, as.character)
            df2$Freq <- as.numeric(df2$Freq)
      
            for (i in seq(from = 1, to = nrow(df2), by = rv_numval)) {
               lower_lim <- i
               upper_lim <- i + rv_numval - 1
               freq <- df2$Freq[lower_lim:upper_lim]
               df2$Freq[lower_lim:upper_lim] <- adjust_values(freq)
      
               if (rv_numval == 2 && identical(tolower(rv_values), c('false','true'))) {
                   # Boolean endogenous variable (single AD for each parent configuration)
                   u_idx <- u_idx + 1
                   head <- paste0(format(round(df2[i + 1,]$Freq, dec_round), nsmall=dec_round, scientific=FALSE), "::u", u_idx, ".")
                   body <- paste0(rv_name, " :- u", u_idx)  # initialize body
                   first_condition <- TRUE
                   for (k in seq(from = 2, to = ncol(df2) - 1, by = 1)) {
                       col_values <- unique(as.character(unlist(df2[, k])))
                       if (!first_condition) {
                           body <- paste0(body, ", ")
                       } else {
                           body <- paste0(body, ", ")
                       }
                       # Boolean formatting: negation if False, atom if True
                       if (identical(tolower(col_values), c('false', 'true'))) {
                           if (identical(tolower(df2[i, k]), 'false')) {
                               body <- paste0(body, "\\+ ", names(df2)[k])
                           } else {
                               body <- paste0(body, names(df2)[k])
                           }
                       } else {
                           # Multivalued parent
                           body <- paste0(body, names(df2)[k], "(", df2[i, k], ")")
                       }
                       first_condition <- FALSE
                   }
                   body <- paste0(body, ".\n")
                   rule <- paste0(head, "\n", body)
                   #print(rule)
                   writeLines(rule, con = output_file)
               } else {
                   # Multivalued or binary non-Boolean endogenous variable
                   u_idx <- u_idx + 1
                   slice_rows <- i:(i + rv_numval - 1)
                   probs <- sapply(slice_rows, function(r) format(round(df2[r,]$Freq, dec_round), nsmall=dec_round, scientific=FALSE))
                   values <- rv_values
                   u_name <- paste0("u", u_idx)
                   ad_parts <- mapply(function(p, v) paste0(p, "::", u_name, "(", v, ")"), probs, values, SIMPLIFY = TRUE)
                   ad_line <- paste(ad_parts, collapse="; ")
                   writeLines(paste0(ad_line, "."), con = output_file)
      
                   # Build mapping rule head
                   body <- paste0(rv_name, "(V) :- ", u_name, "(V)")
                   first_condition <- TRUE
                   # iterate over the conditioning variables
                   for (k in seq(from = 2, to = ncol(df2) - 1, by = 1)) {
                       col_values <- unique(as.character(unlist(df2[, k])))
                       if (!first_condition) {
                           body <- paste0(body, ", ")
                       } else {
                           body <- paste0(body, ", ")
                       }
                       if (length(col_values) > 2) {
                           # multivalued parent
                           # Use df2[i,k] which corresponds to the parent value in this parent configuration
                           body <- paste0(body, names(df2)[k], "(", df2[i, k], ")")
                       } else {
                           if (identical(tolower(col_values[1]), 'false') && identical(tolower(col_values[2]), 'true')) {
                               if (identical(tolower(df2[i, k]), 'false')) {
                                  body <- paste0(body, "\\+ ", names(df2)[k])
                               } else {
                                  body <- paste0(body, names(df2)[k])
                               }
                           } else {
                               body <- paste0(body, names(df2)[k], "(", df2[i, k], ")")
                           }
                       }
                       first_condition <- FALSE
                   }
                   body <- paste0(body, ".\n")
                   rule <- paste0(body)
                   #print(rule)
                   writeLines(rule, con = output_file)
               }
            }
        }
      }

      #writeLines("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%", con = output_file)
      #writeLines("% Integrity constraints ", con = output_file)
      #writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n", con = output_file)

      #actions <- as.character(unique(df_factor$action))
      #if (length(actions) >= 2) {
      #  exclusions <- combn(actions, 2, FUN = function(x) {
      #    sprintf(":- action(%s), action(%s).", x[1], x[2])
      #  })
      #  constraint_rules <- paste(exclusions, collapse = "\n")
      #  writeLines(constraint_rules, con = output_file)
      #}
      close(output_file)

      # Clean memory for fold
      rm(train_dt, train_sample, df_factor, bn_fit, network_structure)
      gc()
    } # end LOOCV folds

    # After finishing all folds for this percentage, append summary numeralia
    numeralia_file <- file.path(cbns_dir, "training_numeralia.txt")
    if (length(training_times) > 0 && length(samples_removed) > 0 && length(train_sample_sizes) > 0) {
      avg_training_time <- mean(training_times)
      sd_training_time <- sd(training_times)
      avg_samples_removed <- mean(samples_removed)
      sd_samples_removed <- sd(samples_removed)
      avg_train_sample_size <- mean(train_sample_sizes)
      sd_train_sample_size <- sd(train_sample_sizes)

      summary_str <- sprintf("\nSummary Statistics:\nAverage Training Time = %.2f seconds, SD = %.2f seconds\nAverage Samples Removed = %.2f, SD = %.2f\nAverage Train Sample Size = %.2f, SD = %.2f\n",
                             avg_training_time, sd_training_time, avg_samples_removed, sd_samples_removed, avg_train_sample_size, sd_train_sample_size)
      write(summary_str, file = numeralia_file, append = TRUE)
    } else {
      write("\nSummary Statistics:\nNo training times, samples removed, or train sample sizes recorded.\n", file = numeralia_file, append = TRUE)
    }

    # End percentage
    message("Completed percentage ", percentage, " for repetition ", rep_num)
  } # end percentages

  # --- Per-repetition summary ---
  rep_summary <- global_summary[Repetition == rep_num]
  message("\nRepetition ", rep_num, " summary:")
  message("Avg training time (s): ", round(mean(rep_summary$TrainingTime_s),2), " ± ", round(sd(rep_summary$TrainingTime_s),2))
  message("Avg samples removed: ", round(mean(rep_summary$SamplesRemoved),1), " ± ", round(sd(rep_summary$SamplesRemoved),1))
  message("Avg train sample size: ", round(mean(rep_summary$TrainSampleSize),1), " ± ", round(sd(rep_summary$TrainSampleSize),1))
  message("\nFinished repetition: ", rep_num, "\n")

} # end repetitions

# --- Save global summary CSV ---
summary_file <- file.path(getwd(),"global_training_summary.csv")
fwrite(global_summary, summary_file)
message("\nGlobal summary written to: ", summary_file)
message("Script finished successfully.")


