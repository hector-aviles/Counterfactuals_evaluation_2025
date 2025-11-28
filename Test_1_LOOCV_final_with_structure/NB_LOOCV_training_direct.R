#!/usr/bin/env Rscript

# ls -l /usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so
# echo "RETICULATE_PYTHON=/usr/bin/python3" >> ~/.Renviron

suppressMessages({
  if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table", repos = "https://cloud.r-project.org")
  if (!requireNamespace("conflicted", quietly = TRUE)) install.packages("conflicted", repos = "https://cloud.r-project.org")
  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", repos = "https://cloud.r-project.org")
  if (!requireNamespace("stringr", quietly = TRUE)) install.packages("stringr", repos = "https://cloud.r-project.org")
  if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr", repos = "https://cloud.r-project.org")
  if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse", repos = "https://cloud.r-project.org")
  if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate", repos = "https://cloud.r-project.org")
})

library(data.table)
library(conflicted)
conflict_prefer("setdiff", "base")
library(dplyr)
library(stringr)
library(tidyr)
library(tidyverse)
library(arules)
library(reticulate)

# Configure Python - use the same path that was detected
#use_python("/usr/bin/python3", required = TRUE)

py_config()

# Check if Python is available and set up reticulate
if (!py_available()) {
  stop("Python is not available. Please ensure Python is installed and configured.")
}

# Source the Python script as a module - make sure it's the correct file
source_python("create_NB_direct.py")

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

# Function to create and train NB model using Python
train_nb_model <- function(train_sample, rep_num, percentage, fold_num, nb_dir) {
  # Convert to data frame with ONLY numerical features (same as original working code)
  # Use only free_* columns which are numerical
  numerical_features <- c("curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W", "latent_collision")
  train_df <- as.data.frame(train_sample)[, c("action", numerical_features)]
  
  # Create model path
  model_path <- file.path(nb_dir, paste0("NB_fold_", fold_num, ".pkl"))
  
  # Use reticulate to call the Python training function
  # Convert R data frame to pandas DataFrame for Python
  pd <- import("pandas")
  train_df_py <- r_to_py(train_df)
  
  # Call the Python function
  result <- create_nb_model_from_data(
    train_df = train_df_py,
    percentage = percentage,
    rep_num = as.integer(rep_num),
    fold_num = as.integer(fold_num),
    model_path = model_path
  )
  
  return(result)
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
dt_no_crashes <- fread(file.path(input_subdir1, "no_crashes.csv"), colClasses = "character")

# Combine crashes and no-crashes into unique examples (LOOCV set)
dt_unique <- rbindlist(list(dt_crashes, dt_no_crashes))
dt_unique[, latent_collision := "True"]

# Add state_id for filtering
state_cols <- c("curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W", "latent_collision")
dt[, state_id := do.call(paste, c(.SD, sep = "_")), .SDcols = state_cols]
dt_unique[, state_id := do.call(paste, c(.SD, sep = "_")), .SDcols = state_cols]

for (nm in c("dt", "dt_unique")) {
  dt_check <- get(nm)
  if (anyNA(dt_check$state_id)) stop("NA values found in state_id in ", nm)
}

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
  dt_rep <- copy(dt)
  dt_unique_rep <- copy(dt_unique)

  # Number of LOOCV folds = number of rows in dt_unique
  n_folds <- nrow(dt_unique_rep)
  message("LOOCV folds (unique examples): ", n_folds)

  for (percentage in percentages) {
    message("\n--- Repetition ", rep_num, " | Percentage ", percentage, " ---\n")

    # Each percentage has its own training_data and NB under the rep folder
    pct_dir <- file.path(rep_dir, percentage)
    training_dir <- file.path(pct_dir, "training_data")
    nb_dir <- file.path(pct_dir, "NB")
    for (d in c(training_dir, nb_dir)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

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

      # --- Create test dataset for the i-th unique example
      current_test_example <- dt_unique_rep[i, ]
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
      current_state_id <- dt_unique_rep$state_id[i]
      num_removed <- sum(dt_rep$state_id == current_state_id)
      samples_removed <- c(samples_removed, num_removed)
      train_dt <- dt_rep[state_id != current_state_id]   # all remaining rows

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

      # --- Train Naive Bayes model using Python
      start_time_nb <- Sys.time()
      
      # Train model directly without saving training file
      nb_result <- train_nb_model(train_sample, rep_num, percentage, i, nb_dir)
      
      end_time_nb <- Sys.time()
      elapsed_time_nb <- as.numeric(end_time_nb - start_time_nb, units = "secs")
      training_times <- c(training_times, elapsed_time_nb)
      
      message(sprintf("NB training completed in %.2f seconds for fold %d", elapsed_time_nb, i))

      # --- Write numeralia (append)
      numeralia_file <- file.path(nb_dir, "training_numeralia.txt")
      numeralia_line <- sprintf("Fold %d: Training Time = %.2f seconds, Samples Removed = %d, Train Sample Size = %d", 
                               i, elapsed_time_nb, num_removed, train_sample_size)
      if (!file.exists(numeralia_file)) {
        writeLines(numeralia_line, numeralia_file)
      } else {
        write(sprintf("\n%s", numeralia_line), file = numeralia_file, append = TRUE)
      }

      # Clean memory for fold
      rm(train_dt, train_sample, dt_test)
      gc()
    } # end LOOCV folds

    # After finishing all folds for this percentage, append summary numeralia
    numeralia_file <- file.path(nb_dir, "training_numeralia.txt")
    if (length(training_times) > 0 && length(samples_removed) > 0 && length(train_sample_sizes) > 0) {
      avg_training_time <- mean(training_times)
      sd_training_time <- sd(training_times)
      avg_samples_removed <- mean(samples_removed)
      sd_samples_removed <- sd(samples_removed)
      avg_train_sample_size <- mean(train_sample_sizes)
      sd_train_sample_size <- sd(train_sample_sizes)

      summary_str <- sprintf("\nSummary Statistics:\nAverage Training Time = %.2f seconds, SD = %.2f seconds\nAverage Samples Removed = %.2f, SD = %.2f\nAverage Train Sample Size = %.2f, SD = %.2f\n",
                             avg_training_time, sd_training_time, avg_samples_removed, sd_samples_removed, 
                             avg_train_sample_size, sd_train_sample_size)
      write(summary_str, file = numeralia_file, append = TRUE)
    } else {
      write("\nSummary Statistics:\nNo training times, samples removed, or train sample sizes recorded.\n", file = numeralia_file, append = TRUE)
    }

    # End percentage
    message("Completed percentage ", percentage, " for repetition ", rep_num)
  } # end percentages

  message("\nFinished repetition: ", rep_num, "\n")

} # end repetitions

message("Script finished successfully.")
