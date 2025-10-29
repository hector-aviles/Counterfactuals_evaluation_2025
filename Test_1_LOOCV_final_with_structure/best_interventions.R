#!/usr/bin/env Rscript

library(dplyr)
library(stringr)

# ----------------------------------------
# Error handling pause function
# ----------------------------------------
pause <- function() {
  while (TRUE) {
    input <- readline(prompt = "An error occurred: Press CTRL-C key to continue (any subsequent code in the script was ignored)")
  }
}
#options(error = pause)

# ----------------------------------------
# Command-line arguments
# ----------------------------------------
args <- commandArgs(trailingOnly = TRUE)

if(length(args) != 2){
  stop("Usage: Rscript merged_best_interventions.R <num_reps> <percentages_comma_separated>")
}

num_reps <- as.numeric(args[1])
percentages <- str_split(args[2], ",")[[1]]

# ----------------------------------------
# Function to check if rows exist in no_crashes (vectorized)
# ----------------------------------------
row_exists <- function(df, ref_df){
  df_str <- apply(df, 1, paste, collapse = "_")
  ref_str <- apply(ref_df, 1, paste, collapse = "_")
  df_str %in% ref_str
}

# ----------------------------------------
# Function to process a single repetition and percentage
# ----------------------------------------
process_rep_perc <- function(rep_num, perc) {
  
  base_dir <- file.path(paste0("rep_", rep_num), perc)
  input_csv <- file.path(base_dir, "cBNs", "twin_networks_results.csv")
  bi_csv <- file.path(base_dir, "cBNs", "best_interventions.csv")
  ds_csv <- file.path(base_dir, "cBNs", "data_sorted.csv")
  ct_csv <- file.path(base_dir, "cBNs", "contingency_table.csv")
  
  no_crashes_csv <- "./Shared_CSVs/no_crashes.csv"

  if(!file.exists(input_csv)){
    cat("Input file does not exist:", input_csv, "\n")
    return(NULL)
  }

  cat("\nProcessing rep:", rep_num, "percentage:", perc, "\n")
  
  # -------------------------
  # Load data
  # -------------------------
  data <- read.csv(input_csv, row.names = NULL)
  no_crashes <- read.csv(no_crashes_csv, row.names = NULL)  
  
  data <- data %>%
    mutate(group_id = rep(1:ceiling(n()/6), each = 6)[1:n()])
  
  data_sorted <- data %>%
    arrange(group_id, probability) %>%
    group_by(group_id) %>%
    mutate(ranking = dense_rank(probability)) %>%
    ungroup()
  
  ds_temp <- data_sorted %>% select(-orig_label_lc, -latent_collision, -iaction, -probability, -elapsed_time, -group_id, -ranking)
  
  data_sorted$potential_crash_before_intervention <- ifelse(row_exists(ds_temp, no_crashes), "False", "True")
  
  ds_temp <- data_sorted %>% select(-orig_label_lc,-latent_collision, -iaction, -probability, -elapsed_time, -group_id, -ranking, -potential_crash_before_intervention)
  ds_temp$action <- data_sorted$iaction
  
  data_sorted$potential_crash_after_intervention <- ifelse(row_exists(ds_temp, no_crashes), "False", "True")
  data_sorted$potential_crash_after_intervention[data_sorted$action == data_sorted$iaction] <- "True"
  data_sorted$best_intervention <- ''
  
  # ----------------------------------------
  # Vectorized best intervention selection
  # ----------------------------------------
  best_candidates <- data_sorted %>% filter(ranking == 1)
  best_selected <- best_candidates %>%
    group_by(group_id) %>%
    slice_sample(n = 1) %>%
    ungroup()
  
  best_indices <- which(paste(data_sorted$group_id, data_sorted$iaction) %in% 
                        paste(best_selected$group_id, best_selected$iaction))
  data_sorted$best_intervention[best_indices] <- '*'
  
  best_int <- best_selected %>% select(-group_id, -ranking, -elapsed_time, -best_intervention)
  
  # Save outputs
  write.csv(best_int, bi_csv, row.names = FALSE)
  cat("Saved best interventions to", bi_csv, "\n")
  
  write.csv(data_sorted, ds_csv, row.names = FALSE)
  cat("Saved sorted data to", ds_csv, "\n")
  
  # -------------------------
  # Vectorized contingency table
  # -------------------------
  crash_data <- data_sorted %>% filter(potential_crash_before_intervention == "True")
  no_crash_data <- data_sorted %>% filter(potential_crash_before_intervention == "False")
  
  output_file <- file(ct_csv, "w")
  
  writeLines("=== CRASH SECTION ===", con = output_file)
  writeLines("(Data from rows where potential_crash_before_intervention is True)\n", con = output_file)
  
  crash_group_counts <- crash_data %>% 
    filter(ranking == 1) %>%
    count(group_id) %>%
    count(n) %>%
    rename(num_best = n, num_groups = nn)
  
  for(i in 1:6){
    num <- crash_group_counts %>% filter(num_best == i) %>% pull(num_groups)
    if(length(num) == 0) num <- 0
    writeLines(paste("Number of groups with exactly", i, "best interventions:", num), con = output_file)
  }
  
  action_counts_crash <- crash_data %>% filter(ranking == 1) %>%
    group_by(iaction) %>%
    summarize(total_selected=n(), safe_count=sum(potential_crash_after_intervention=="False"),
              unsafe_count=sum(potential_crash_after_intervention=="True"))
  
  writeLines("\nNumber of actions selected for each intervention type:", con = output_file)
  for(i in 1:nrow(action_counts_crash)){
    writeLines(paste(action_counts_crash$iaction[i], ": Selected", action_counts_crash$total_selected[i],
                     "times, Safe", action_counts_crash$safe_count[i],
                     "times, Unsafe", action_counts_crash$unsafe_count[i], "times"), con = output_file)
  }
  
  writeLines("\n=== NO CRASH SECTION ===", con = output_file)
  writeLines("(Data from rows where potential_crash_before_intervention is False)\n", con = output_file)
  
  no_crash_group_counts <- no_crash_data %>% 
    filter(ranking == 1) %>%
    count(group_id) %>%
    count(n) %>%
    rename(num_best = n, num_groups = nn)
  
  for(i in 1:6){
    num <- no_crash_group_counts %>% filter(num_best == i) %>% pull(num_groups)
    if(length(num) == 0) num <- 0
    writeLines(paste("Number of groups with exactly", i, "best interventions:", num), con = output_file)
  }
  
  close(output_file)
  cat("Saved contingency table to", ct_csv, "\n")
}

# ----------------------------------------
# Loop through repetitions and percentages
# ----------------------------------------
for(rep in 1:num_reps){
  for(perc in percentages){
    process_rep_perc(rep, perc)
  }
}

cat("\nAll repetitions and percentages processed successfully!\n")

