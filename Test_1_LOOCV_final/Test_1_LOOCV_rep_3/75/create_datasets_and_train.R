library(data.table)
#install.packages("conflicted")
library(conflicted)  
conflict_prefer("setdiff", "base")  # Resolve setdiff conflict
#install.packages("bnlearn")
library(bnlearn)
#install.packages("dplyr")
library(dplyr)
#install.packages("stringr")
library(stringr)
#install.packages("tidyr")
library(tidyr)
# There are conflicts between tidyverse an dplyr
# https://stackoverflow.com/questions/73336628/package-conflicts-in-r
#install.packages("tidyverse")
library(tidyverse)
conflict_prefer("filter", "dplyr")
conflict_prefer("lag", "dplyr")
# Required to use discretization function
#install.packages("arules")
library("arules")

# Define our error-handling pause function:
pause = function(){
    while (TRUE) {
        input <- readline(prompt="An error ocurred: Press CTRL-C key to continue (any subsequent code in the script was  ignored)")
    }    
}

##############
## Normalize probability values
##############
adjust_values <- function(freq) {
   # Remove NAs
   freq[is.na(freq)] <- 0.0
   # It seems that WhatIf has problems using small values     
   freq[ freq <= 0.0001 ] <- 0.001
   total_sum = sum(freq)
   if (total_sum == 0.0) {
      # if there are no values greater than 0, set a uniform distribution
      freq <- 1 / length(freq)
   } else {        
      # If the sum is greater than 1, normalize the values to sum to 1
      if (total_sum > 1) {
        freq <- freq / total_sum
      } 
   }  
   return(freq)
}

# Set the 'error' option to execute our pause function:
options(error=pause)

# Set seed for reproducibility
set.seed(300)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
output_subdir1 <- "./data_split/"
input_subdir1 <- "../Shared_CSVs/"
percentage <- "01"

if (length(args) > 0) {
  output_subdir1 <- args[1]
  input_subdir1 <- args[2]
  percentage <- args[3]
} else {
  message("No parameters provided (requires output_subdir1, input_subdir1, and percentage).")
}

message("output_subdir1: ", output_subdir1)
message("input_subdir1: ", input_subdir1)
message("percentage: ", percentage)

# Validate input files
input_files <- file.path(input_subdir1, c("complete_DB_discrete.csv", "crashes.csv", "no_crashes.csv"))
for (f in input_files) {
  if (!file.exists(f)) stop("Input file missing: ", f)
}

# Read datasets with fread
df_path <- file.path(input_subdir1, "complete_DB_discrete.csv")
dt <- fread(df_path, colClasses = "character")
message("Complete dataset dimensions: ", paste(dim(dt), collapse = " x "))

dt <- dt[complete.cases(dt), ]
dim(dt)

crashes_path <- file.path(input_subdir1, "crashes.csv")
dt_crashes <- fread(crashes_path, colClasses = "character")
dt_crashes[, labeled_lc := "True"]
message("Crash dataset dimensions: ", paste(dim(dt_crashes), collapse = " x "))

no_crashes_path <- file.path(input_subdir1, "no_crashes.csv")
dt_no_crashes <- fread(no_crashes_path, colClasses = "character")
dt_no_crashes[, labeled_lc := "False"]
message("No crash dataset dimensions: ", paste(dim(dt_no_crashes), collapse = " x "))

# Combine crashes and no-crashes
dt_unique <- rbindlist(list(dt_crashes, dt_no_crashes))
dt_unique[, latent_collision := "True"]

# Add state_id for internal filtering
state_cols <- c("curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W")
dt[, state_id := do.call(paste, c(.SD, sep = "_")), .SDcols = state_cols]
dt_unique[, state_id := do.call(paste, c(.SD, sep = "_")), .SDcols = state_cols]

# Validate state_id construction
for (dt_name in c("dt", "dt_unique")) {
  dt_to_check <- get(dt_name)
  if (anyNA(dt_to_check$state_id)) {
    stop("NA values found in state_id in ", dt_name)
  }
}

# Sampling fraction
fraction <- as.numeric(percentage) / 100
if (fraction <= 0 | fraction > 1) stop("Percentage must be between 0 and 100")

# Ensure training_data and test_data directories exist
train_dir <- file.path(output_subdir1, "training_data")
test_dir <- file.path(output_subdir1, "test_data")
for (d in c(train_dir, test_dir)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# Initialize lists to store training times, samples removed, and train sample sizes
training_times <- numeric()
samples_removed <- numeric()
train_sample_sizes <- numeric()

# Loop over nrow(dt_unique) for LOOCV
for (i in 1:nrow(dt_unique)) {
  message("Processing row ", i, " of ", nrow(dt_unique))
  
  # Create test dataset
  current_test_example <- dt_unique[i, ]
  message("Creating test dataset...")
  action_list <- c("change_to_left", "change_to_right", "cruise", "keep", "swerve_left", "swerve_right")
  dt_test <- current_test_example[rep(1:.N, each = 6)][, iaction := rep(action_list, times = nrow(current_test_example))]
  message("Done.")
  message("df_test dimensions: ", paste(dim(dt_test), collapse = " x "))  
  for (dt_name in c("dt_test")) {
    dt_to_check <- get(dt_name)
    if (anyNA(dt_to_check$state_id)) {
      stop("NA values found in state_id in ", dt_name)
    }
  }  
  
  # Extract state_id for the i-th row
  current_state_id <- dt_unique$state_id[i]
  
  # Remove all examples with matching state from dt and record number removed
  num_removed <- sum(dt$state_id == current_state_id)
  samples_removed <- c(samples_removed, num_removed)
  train_dt <- dt[state_id != current_state_id]
  
  # Validate no state overlap
  if (current_state_id %in% train_dt$state_id) {
    stop("Data leakage detected: test state_id found in training set for fold ", i)
  }
  
  # Sample with at least one per action if possible
  action_list <- unique(train_dt$action)
  num_actions <- length(action_list)
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
  
  # Record train sample size
  train_sample_size <- nrow(train_sample)
  train_sample_sizes <- c(train_sample_sizes, train_sample_size)
  
  # Display action distribution
  action_dist <- train_sample[, .N, by = action]
  message("Action distribution for fold ", i, ":\n", paste(capture.output(print(action_dist)), collapse = "\n"))
  
  # Convert all columns to character and exclude state_id
  train_sample <- train_sample[, lapply(.SD, as.character), .SDcols = setdiff(names(train_sample), "state_id")]
  
  # Save dt_test after converting to character and excluding state_id
  dt_test <- dt_test[, lapply(.SD, as.character), .SDcols = setdiff(names(dt_test), "state_id")]
  test_file <- file.path(test_dir, paste0("test_fold_", i, ".csv"))
  fwrite(dt_test, test_file, quote = TRUE)
  
  # Train the model
  output <- paste0("./cBNs/cBN_", i, ".pl")
  
  bl_a <- data.frame(from = c("action", "action", "action", "action", "action", "action", "action"), to = c("curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W"))
  bl_cl <- data.frame(from = c("curr_lane", "curr_lane", "curr_lane", "curr_lane", "curr_lane", "curr_lane"), to = c("free_E","free_NE","free_NW","free_SE","free_SW","free_W"))
  bl_e <- data.frame(from = c("free_E", "free_E", "free_E", "free_E", "free_E", "free_E"), to = c("curr_lane","free_NE","free_NW","free_SE","free_SW","free_W"))
  bl_ne <- data.frame(from = c("free_NE", "free_NE", "free_NE", "free_NE", "free_NE", "free_NE"), to = c("curr_lane","free_E","free_NW","free_SE","free_SW","free_W"))
  bl_nw <- data.frame(from = c("free_NW", "free_NW", "free_NW", "free_NW", "free_NW", "free_NW"), to = c("curr_lane","free_E","free_NE","free_SE","free_SW","free_W"))
  bl_se <- data.frame(from = c("free_SE", "free_SE", "free_SE", "free_SE", "free_SE", "free_SE"), to = c("curr_lane","free_E","free_NE","free_NW","free_SW","free_W"))
  bl_sw <- data.frame(from = c("free_SW", "free_SW", "free_SW", "free_SW", "free_SW", "free_SW"), to = c("curr_lane","free_E","free_NE","free_NW","free_SE","free_W"))
  bl_w <- data.frame(from = c("free_W", "free_W", "free_W", "free_W", "free_W", "free_W"), to = c("curr_lane","free_E","free_NE","free_NW","free_SE","free_SW"))
  bl_lcol <- data.frame(from = c("latent_collision", "latent_collision", "latent_collision", "latent_collision", "latent_collision", "latent_collision", "latent_collision", "latent_collision"), to = c("action",  "curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W"))
  bl <- rbind(bl_a, bl_cl, bl_e, bl_ne, bl_nw, bl_se, bl_sw, bl_w, bl_lcol)
  
  # Convert variables to factors
  df_factor <- as.data.frame(lapply(train_sample, factor))
  
  # Record training time for structural learning
  start_time_struct <- Sys.time()
  network_structure <- hc(df_factor, whitelist = NULL, blacklist = bl, score = "bic", debug = FALSE, restart = 0, perturb = 1, max.iter = Inf, maxp = Inf, optimized = TRUE)
  end_time_struct <- Sys.time()
  elapsed_time_struct <- as.numeric(end_time_struct - start_time_struct, units = "secs")
  
  model_filename <- paste(output, ".dot", sep = "", collapse = "")
  write.dot(network_structure, file = model_filename)
  model_filename <- paste(output, ".png", sep = "", collapse = "")
  png(model_filename)
  plot(network_structure)
  dev.off()
  
  print(network_structure)
  
  cmd <- paste("dot -Tps ", output, ".dot  -o ", output, ".ps", sep = "", collapse = "")
  system(cmd, intern = TRUE)
  
  # Record training time for parametric learning
  start_time_param <- Sys.time()
  bn_fit <- bn.fit(network_structure, data = df_factor, method = "mle", replace.unidentifiable = TRUE)
  end_time_param <- Sys.time()
  elapsed_time_param <- as.numeric(end_time_param - start_time_param, units = "secs")
  
  # Total training time for this fold
  total_training_time <- elapsed_time_struct + elapsed_time_param
  training_times <- c(training_times, total_training_time)
  
  # Write training times, samples removed, and train sample size to file
  numeralia_file <- "./cBNs/training_numeralia.txt"
  if (i == 1) {
    # Create or overwrite file for the first iteration
    writeLines(sprintf("Fold %d: Training Time = %.2f seconds, Samples Removed = %d, Train Sample Size = %d", 
                       i, total_training_time, num_removed, train_sample_size), numeralia_file)
  } else {
    # Append to file for subsequent iterations
    write(sprintf("\nFold %d: Training Time = %.2f seconds, Samples Removed = %d, Train Sample Size = %d", 
                  i, total_training_time, num_removed, train_sample_size), file = numeralia_file, append = TRUE)
  }
  
  # Write parameters to .net file
  model_filename <- paste(output, ".net", sep = "", collapse = "")
  write.net(model_filename, bn_fit)
  
  output_file <- file(output, "w")
  writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%", con = output_file)  
  writeLines("% Probabilistic facts", con = output_file)  
  writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n", con = output_file)  
  
  # Extract facts
  u_idx <- 0
  for (rv in bn_fit) {
    rv_name <- rv$node
    rv_values <- dimnames(rv$prob)[[1]]   
    rv_numval = length(rv_values) 
         
    if (identical(rv$parents, character(0))){ 
        print("Processing RV: ")
        flush.console()
        print(rv_name)
        df2 <- as.data.frame(rv$prob)
        df2[] <- lapply(df2, as.character)
        df2$Freq <- as.numeric(df2$Freq)
        freq <- adjust_values(df2$Freq)
        df2$Freq <- freq
  
        if (rv_numval == 2 && identical(tolower(rv_values[1]), 'false') && identical(tolower(rv_values[2]), 'true')) {
            u_idx <- u_idx + 1
            fact <- paste(format(df2[2,]$Freq, nsmall=6, scientific=FALSE), "::u", u_idx, ".\n", 
                          rv_name, " :- u", u_idx, ".", sep = "", collapse = "")
            print(fact)
            writeLines(fact, con = output_file)
        } else {
            for (i in seq(from = 1, to = rv_numval, by = 1)) {
                u_idx <- u_idx + 1
                fact <- paste(format(df2[i,]$Freq, nsmall=6, scientific=FALSE), "::u", u_idx, ".", sep = "", collapse = "")
                body <- paste(rv_name, "(", df2[i,1], ")", sep = "", collapse = "")
                for (j in seq(from = 1, to = rv_numval, by = 1)) {
                    if (j != i) {
                        body <- paste(body, ", \\+ ", rv_name, "(", df2[j,1], ")", sep = "", collapse = "")
                    }
                }
                body <- paste(body, ", u", u_idx, ".", sep = "", collapse = "")
                fact <- paste(fact, "\n", body, sep = "", collapse = "")
                print(fact)
                writeLines(fact, con = output_file)
            }
        }
    }
  }
  
  writeLines("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%", con = output_file)  
  writeLines("% Rules", con = output_file)  
  writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n", con = output_file)
  
  # Extract rules
  for (rv in bn_fit) {
    rv_name <- rv$node
    rv_values <- dimnames(rv$prob)[[1]]   
    rv_numval = length(rv_values) 
  
    if (!identical(rv$parents, character(0))){ 
        print("Processing RV: ")
        flush.console()
        print(rv_name)
        df2 <- as.data.frame(rv$prob)
        df2[] <- lapply(df2, as.character)
        df2$Freq <- as.numeric(df2$Freq)
    
        for (i in seq(from = 1, to = nrow(df2), by = rv_numval)) {
           lower_lim <- i
           upper_lim <- i + rv_numval - 1
           freq <- df2$Freq[lower_lim:upper_lim]
           df2$Freq[lower_lim:upper_lim] <- adjust_values(freq) 
  
           if (rv_numval == 2 && identical(tolower(rv_values[1]), 'false') && identical(tolower(rv_values[2]), 'true')) {
               u_idx <- u_idx + 1
               head <- paste(format(df2[i + 1,]$Freq, nsmall=6, scientific=FALSE), "::u", u_idx, ".", sep = "", collapse = "")
               body <- paste(rv_name, " :- ", sep = "", collapse = "")
               first_condition <- TRUE
               for (k in seq(from = 2, to = ncol(df2) - 1, by = 1)) {
                   col_values <- unique(as.character(unlist(df2[, k])))
                   if (!first_condition) {
                       body <- paste(body, ", ", sep = "", collapse = "")
                   }
                   if (length(col_values) > 2) {
                       body <- paste(body, names(df2)[k], "(", df2[i, k], ")", sep = "", collapse = "")
                   } else {
                       if (identical(tolower(col_values[1]), 'false') && identical(tolower(col_values[2]), 'true')) {
                           if (identical(tolower(df2[i, k]), 'false')) {
                               body <- paste(body, "\\+ ", names(df2)[k], sep = "", collapse = "")
                           } else {
                               body <- paste(body, names(df2)[k], sep = "", collapse = "")
                           }
                       } else {
                           body <- paste(body, names(df2)[k], "(", df2[i, k], ")", sep = "", collapse = "")
                       }
                   }
                   first_condition <- FALSE
               }
               body <- paste(body, ", u", u_idx, ".", sep = "", collapse = "")
               rule <- paste(head, "\n", body, "\n", sep = "", collapse = "")
               print(rule)
               writeLines(rule, con = output_file)
           } else {
               for (j in seq(from = i, to = i + rv_numval - 1, by = 1)) {
                   u_idx <- u_idx + 1
                   head <- paste(format(df2[j,]$Freq, nsmall=6, scientific=FALSE), "::u", u_idx, ".", sep = "", collapse = "")
                   body <- paste(rv_name, "(", df2[j,1], ") :- ", sep = "", collapse = "")
                   first_condition <- TRUE
                   for (k in seq(from = 2, to = ncol(df2) - 1, by = 1)) {
                       col_values <- unique(as.character(unlist(df2[, k])))
                       if (!first_condition) {
                           body <- paste(body, ", ", sep = "", collapse = "")
                       }
                       if (length(col_values) > 2) {
                           body <- paste(body, names(df2)[k], "(", df2[j, k], ")", sep = "", collapse = "")
                       } else {
                           if (identical(tolower(col_values[1]), 'false') && identical(tolower(col_values[2]), 'true')) {
                               if (identical(tolower(df2[j, k]), 'false')) {
                                   body <- paste(body, "\\+ ", names(df2)[k], sep = "", collapse = "")
                               } else {
                                   body <- paste(body, names(df2)[k], sep = "", collapse = "")
                               }
                           } else {
                               body <- paste(body, names(df2)[k], "(", df2[j, k], ")", sep = "", collapse = "")
                           }
                       }
                       first_condition <- FALSE
                   }
                   body <- paste(body, ", u", u_idx, ".", sep = "", collapse = "")
                   rule <- paste(head, "\n", body, "\n", sep = "", collapse = "")
                   print(rule)
                   writeLines(rule, con = output_file)
               }
           }
        }
    }
  }
  
  writeLines("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%", con = output_file)  
  writeLines("% Integrity constraints ", con = output_file)  
  writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n", con = output_file)
  
  actions <- as.character(unique(df_factor$action))
  exclusions <- combn(actions, 2, FUN = function(x) {
    sprintf(":- action(%s), action(%s).", x[1], x[2])
  })
  constraint_rules <- paste(exclusions, collapse = "\n")
  writeLines(constraint_rules, con = output_file)
  close(output_file)  
  
  # Clean up memory
  rm(train_dt, train_sample, action_dist, dt_test, current_test_example)
  gc()
}

# Calculate and append statistics after the loop
numeralia_file <- "./cBNs/training_numeralia.txt"
if (length(training_times) > 0 && length(samples_removed) > 0 && length(train_sample_sizes) > 0) {
  avg_training_time <- mean(training_times)
  sd_training_time <- sd(training_times)
  avg_samples_removed <- mean(samples_removed)
  sd_samples_removed <- sd(samples_removed)
  avg_train_sample_size <- mean(train_sample_sizes)
  sd_train_sample_size <- sd(train_sample_sizes)
  
  write(sprintf("\nSummary Statistics:\nAverage Training Time = %.2f seconds, SD = %.2f seconds\nAverage Samples Removed = %.2f, SD = %.2f\nAverage Train Sample Size = %.2f, SD = %.2f", avg_training_time, sd_training_time, avg_samples_removed, sd_samples_removed, avg_train_sample_size, sd_train_sample_size), file = numeralia_file, append = TRUE)
} else {
  write("\nSummary Statistics:\nNo training times, samples removed, or train sample sizes recorded.", file = numeralia_file, append = TRUE)
}

# Clean up memory
rm(dt, dt_crashes, dt_no_crashes, dt_unique)
gc()

message("Processing complete")

options(error=NULL)
