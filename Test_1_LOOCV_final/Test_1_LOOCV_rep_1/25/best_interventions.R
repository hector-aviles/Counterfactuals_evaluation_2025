library(dplyr)

# Define our error-handling pause function
pause = function(){
    while (TRUE) {
        input <- readline(prompt="An error occurred: Press CTRL-C key to continue (any subsequent code in the script was ignored)")
    }    
}

# Set the 'error' option to execute our pause function
options(error=pause)

args <- commandArgs(trailingOnly = TRUE)

input1_csv <- "TNs_results.csv"
output1_csv <- "best_interventions.csv"
input2_csv <- "../no_crashes.csv"
output2_csv <- "../data_sorted.csv"
output3_csv <- "../contingency_table.csv"

# Check if optional parameter is provided
if (length(args) > 0) {
    input1_csv <- args[1]
    output1_csv <- args[2]
    input2_csv <- args[3]
    output2_csv <- args[4]
    output3_csv <- args[5]    
} else {
    print("No parameters provided (it requires one csv file and the output file).")
}

print(paste("Input CSV: ", input1_csv))
print(paste("Output CSV: ", output1_csv))  

# Read the CSV file
data <- read.csv(input1_csv)
dim(data)

no_crashes <- read.csv(input2_csv)
dim(data)

# Add row numbers to track groups of 6 rows (1 group for each 6 rows)
data <- data %>%
  mutate(group_id = rep(1:(n()/6), each = 6))  # Assuming the rows are already sorted in groups of 6

# Sort the data within each group_id from minimum to maximum probability
data_sorted <- data %>%
  arrange(group_id, probability)

# Add ranking within each group of 6 rows based on probability
data_sorted <- data_sorted %>%
  group_by(group_id) %>%
  mutate(ranking = dense_rank(probability)) %>%  # Rank probabilities within each group, with ties
  ungroup()
  
ds_temp <- data_sorted %>% select(-labeled_lc, -latent_collision, -iaction, -probability, -elapsed_time, -group_id, -ranking)
names(ds_temp)

data_sorted$potential_crash_before_intervention <- 'True'

print("Check for each row in data_sorted if it exists in no_crashes...")

# Check for each row in data_sorted if it exists in no_crashes
for (i in 1:nrow(ds_temp)) {
  # Check if the row in 'data_sorted' exists in 'no_crashes'
  if (any(apply(no_crashes, 1, function(row) all(row == ds_temp[i, ])))) {
    data_sorted$potential_crash_before_intervention[i] <- 'False'
  }
}

ds_temp <- data_sorted %>% select(-labeled_lc,-latent_collision, -iaction, -probability, -elapsed_time, -group_id, -ranking, -potential_crash_before_intervention)
names(ds_temp)

ds_temp$action <- data_sorted$iaction 

data_sorted$potential_crash_after_intervention <- 'True'

print("Check for each row in data_sorted if it exists in no_crashes...")
# Check for each row in data_sorted if it exists in no_crashes
for (i in 1:nrow(ds_temp)) {
  # Check if the row in 'data_sorted' exists in 'no_crashes'
  if (any(apply(no_crashes, 1, function(row) all(row == ds_temp[i, ])))) {
    data_sorted$potential_crash_after_intervention[i] <- 'False'
  }
}

# Assign 'True' to data_sorted$potential_crash_after_intervention, in every row where  data_sorted$action == data_sorted$iaction  
data_sorted$potential_crash_after_intervention[data_sorted$action == data_sorted$iaction] <- "True"
  
data_sorted$best_intervention <- ''  

# Initialize a vector to count groups with 1, 2, 3, 4, 5, or 6 rows with ranking 1
counting <- c(0, 0, 0, 0, 0, 0)

# Initialize an empty dataframe to store the best interventions
best_int <- data.frame()

# Use a for loop to go through each group and select the best intervention
for(group in unique(data_sorted$group_id)) {
  
  # Subset data for the current group
  group_data <- data_sorted %>% filter(group_id == group)
  
  # Find all actions with ranking 1
  best_candidates <- group_data %>% filter(ranking == 1)
  
  # Count the number of rows with ranking 1 in this group
  num_best <- nrow(best_candidates)
  
  # Increment the corresponding count in the counting vector
  counting[num_best] <- counting[num_best] + 1
  
  # If there is only one action with rank 1, select it
  if(num_best == 1) {
    best_intervention <- best_candidates
  } 
  # If there are two or more, randomly select one
  else if(num_best > 1) {
    best_intervention <- best_candidates[sample(nrow(best_candidates), 1), ]
  }
  
  # Mark the best action in the data_sorted by adding '*' in the 'best_intervention' column
  data_sorted$best_intervention[data_sorted$group_id == group & data_sorted$iaction == best_intervention$iaction] <- '*'
  
  # Add the selected best action to the best_int dataframe
  best_int <- rbind(best_int, best_intervention)
}

# Select only necessary columns if needed (e.g., remove group_id)
best_int <- best_int %>% select(-group_id, -ranking, -elapsed_time, -best_intervention)

# Save the result to a new CSV file
write.csv(best_int, output1_csv, row.names = FALSE)
# Print confirmation
cat("The best interventions have been saved to ", output1_csv)

# Save the result to a new CSV file
write.csv(data_sorted, output2_csv, row.names = FALSE)
# Print confirmation
cat("Data sorted have been saved to ", output2_csv)

##########

# First, create the filtered datasets
crash_data <- data_sorted %>% filter(potential_crash_before_intervention == "True")
no_crash_data <- data_sorted %>% filter(potential_crash_before_intervention == "False")

# Open the output file
output_file <- file(output3_csv, "w")

### CRASH SECTION ###
writeLines("=== CRASH SECTION ===", con = output_file)
writeLines("(Data from rows where potential_crash_before_intervention is True)\n", con = output_file)

# Counting results for crash section
for (i in 1:length(counting)) {
   str <- paste("Number of groups with exactly ", i, " best interventions: ", 
               sum(crash_data %>% filter(ranking == 1) %>% count(group_id) %>% pull(n) == i))
   writeLines(str, con = output_file)
}

# Action counts for crash section
action_counts_crash <- crash_data %>%
  filter(ranking == 1) %>%
  group_by(iaction) %>%
  summarize(
    total_selected = n(),
    safe_count = sum(potential_crash_after_intervention == "False"),
    unsafe_count = sum(potential_crash_after_intervention == "True")
  )

writeLines("\nNumber of actions selected for each intervention type:", con = output_file)
for (i in 1:nrow(action_counts_crash)) {
  str <- paste(action_counts_crash$iaction[i], ": Selected", action_counts_crash$total_selected[i], 
               "times, Safe", action_counts_crash$safe_count[i], 
               "times, Unsafe", action_counts_crash$unsafe_count[i], "times")
  writeLines(str, con = output_file)
}

# Totals for crash section
total_safe_crash <- sum(crash_data %>% filter(ranking == 1) %>% pull(potential_crash_after_intervention) == "False")
total_unsafe_crash <- sum(crash_data %>% filter(ranking == 1) %>% pull(potential_crash_after_intervention) == "True")

writeLines("\nTotal safe/unsafe actions:", con = output_file)
writeLines(paste("Total safe actions:", total_safe_crash), con = output_file)
writeLines(paste("Total unsafe actions:", total_unsafe_crash), con = output_file)

# Transition matrix for crash section
writeLines("\nTransition matrix (before -> after):", con = output_file)
transition_crash <- with(crash_data %>% filter(ranking == 1),
                         table(factor(potential_crash_before_intervention, levels = c("True", "False")),
                               factor(potential_crash_after_intervention, levels = c("True", "False"))))

writeLines(paste("    True -> True:", 
                transition_crash["True", "True"],
                "(from", crash_data %>% filter(ranking == 1, potential_crash_before_intervention == "True", 
                                             potential_crash_after_intervention == "True") %>% 
                  distinct(group_id) %>% nrow(), "groups)"), 
           con = output_file)
writeLines(paste("    True -> False:", 
                transition_crash["True", "False"],
                "(from", crash_data %>% filter(ranking == 1, potential_crash_before_intervention == "True", 
                                             potential_crash_after_intervention == "False") %>% 
                  distinct(group_id) %>% nrow(), "groups)"), 
           con = output_file)
           
# Detailed transition matrices by number of ties for CRASH section
writeLines("\nDetailed transition matrices by number of ties (CRASH SECTION):", con = output_file)
for (i in 1:6) {
  # Only process if we have groups with this number of ties
  if (sum(crash_data %>% filter(ranking == 1) %>% count(group_id) %>% pull(n) == i) > 0) {
    tied_data <- crash_data %>%
      filter(ranking == 1) %>%
      group_by(group_id) %>%
      filter(n() == i)  # Groups with exactly i ties
    
    writeLines(paste("\nFor groups with", i, "tied best interventions:"), con = output_file)
    writeLines("  - Transition matrix (before -> after):", con = output_file)
    
    # Create transition matrix
    transition <- with(tied_data,
                      table(factor(potential_crash_before_intervention, levels = c("True", "False")),
                            factor(potential_crash_after_intervention, levels = c("True", "False"))))
    
    # Get group counts
    group_counts <- tied_data %>%
      group_by(Before = potential_crash_before_intervention,
               After = potential_crash_after_intervention) %>%
      summarize(n_groups = n_distinct(group_id), .groups = "drop")
    
    # Write all combinations
    writeLines(paste("    True -> True:", 
                    transition["True", "True"],
                    "(from", group_counts %>% filter(Before == "True", After == "True") %>% pull(n_groups) %>% ifelse(length(.) == 0, 0, .),
                    "groups)"), 
               con = output_file)
    writeLines(paste("    True -> False:", 
                    transition["True", "False"],
                    "(from", group_counts %>% filter(Before == "True", After == "False") %>% pull(n_groups) %>% ifelse(length(.) == 0, 0, .),
                    "groups)"), 
               con = output_file)
  }
}           

### NO CRASH SECTION ###
writeLines("\n\n=== NO CRASH SECTION ===", con = output_file)
writeLines("(Data from rows where potential_crash_before_intervention is False)\n", con = output_file)

# Counting results for no_crash section
for (i in 1:length(counting)) {
   str <- paste("Number of groups with exactly ", i, " best interventions: ", 
               sum(no_crash_data %>% filter(ranking == 1) %>% count(group_id) %>% pull(n) == i))
   writeLines(str, con = output_file)
}

# Action counts for no_crash section
action_counts_no_crash <- no_crash_data %>%
  filter(ranking == 1) %>%
  group_by(iaction) %>%
  summarize(
    total_selected = n(),
    safe_count = sum(potential_crash_after_intervention == "False"),
    unsafe_count = sum(potential_crash_after_intervention == "True")
  )

writeLines("\nNumber of actions selected for each intervention type:", con = output_file)
for (i in 1:nrow(action_counts_no_crash)) {
  str <- paste(action_counts_no_crash$iaction[i], ": Selected", action_counts_no_crash$total_selected[i], 
               "times, Safe", action_counts_no_crash$safe_count[i], 
               "times, Unsafe", action_counts_no_crash$unsafe_count[i], "times")
  writeLines(str, con = output_file)
}

# Totals for no_crash section
total_safe_no_crash <- sum(no_crash_data %>% filter(ranking == 1) %>% pull(potential_crash_after_intervention) == "False")
total_unsafe_no_crash <- sum(no_crash_data %>% filter(ranking == 1) %>% pull(potential_crash_after_intervention) == "True")

writeLines("\nTotal safe/unsafe actions:", con = output_file)
writeLines(paste("Total safe actions:", total_safe_no_crash), con = output_file)
writeLines(paste("Total unsafe actions:", total_unsafe_no_crash), con = output_file)

# Transition matrix for no_crash section
writeLines("\nTransition matrix (before -> after):", con = output_file)
transition_no_crash <- with(no_crash_data %>% filter(ranking == 1),
                         table(factor(potential_crash_before_intervention, levels = c("True", "False")),
                               factor(potential_crash_after_intervention, levels = c("True", "False"))))

writeLines(paste("    False -> True:", 
                transition_no_crash["False", "True"],
                "(from", no_crash_data %>% filter(ranking == 1, potential_crash_before_intervention == "False", 
                                                 potential_crash_after_intervention == "True") %>% 
                  distinct(group_id) %>% nrow(), "groups)"), 
           con = output_file)
writeLines(paste("    False -> False:", 
                transition_no_crash["False", "False"],
                "(from", no_crash_data %>% filter(ranking == 1, potential_crash_before_intervention == "False", 
                                                 potential_crash_after_intervention == "False") %>% 
                  distinct(group_id) %>% nrow(), "groups)"), 
           con = output_file)
           
# Detailed transition matrices by number of ties for NO CRASH section
writeLines("\n\nDetailed transition matrices by number of ties (NO CRASH SECTION):", con = output_file)
for (i in 1:6) {
  # Only process if we have groups with this number of ties
  if (sum(no_crash_data %>% filter(ranking == 1) %>% count(group_id) %>% pull(n) == i) > 0) {
    tied_data <- no_crash_data %>%
      filter(ranking == 1) %>%
      group_by(group_id) %>%
      filter(n() == i)  # Groups with exactly i ties
    
    writeLines(paste("\nFor groups with", i, "tied best interventions:"), con = output_file)
    writeLines("  - Transition matrix (before -> after):", con = output_file)
    
    # Create transition matrix
    transition <- with(tied_data,
                      table(factor(potential_crash_before_intervention, levels = c("True", "False")),
                            factor(potential_crash_after_intervention, levels = c("True", "False"))))
    
    # Get group counts
    group_counts <- tied_data %>%
      group_by(Before = potential_crash_before_intervention,
               After = potential_crash_after_intervention) %>%
      summarize(n_groups = n_distinct(group_id), .groups = "drop")
    
    # Write all combinations
    writeLines(paste("    False -> True:", 
                    transition["False", "True"],
                    "(from", group_counts %>% filter(Before == "False", After == "True") %>% pull(n_groups) %>% ifelse(length(.) == 0, 0, .),
                    "groups)"), 
               con = output_file)
    writeLines(paste("    False -> False:", 
                    transition["False", "False"],
                    "(from", group_counts %>% filter(Before == "False", After == "False") %>% pull(n_groups) %>% ifelse(length(.) == 0, 0, .),
                    "groups)"), 
               con = output_file)
  }
}

# Final summary section
writeLines("\n\n=== FINAL SUMMARY ===", con = output_file)

# Calculate the random selection outcomes from best_int
random_true_true <- sum(best_int$potential_crash_before_intervention == "True" & 
                        best_int$potential_crash_after_intervention == "True")
random_false_false <- sum(best_int$potential_crash_before_intervention == "False" & 
                          best_int$potential_crash_after_intervention == "False")
random_true_false <- sum(best_int$potential_crash_before_intervention == "True" & 
                         best_int$potential_crash_after_intervention == "False")
random_false_true <- sum(best_int$potential_crash_before_intervention == "False" & 
                         best_int$potential_crash_after_intervention == "True")

writeLines(paste("Random selection - Crash before (True) and after intervention (True):", random_true_true), con = output_file)
writeLines(paste("Random selection - Crash before (False) and after intervention (False):", random_false_false), con = output_file)
writeLines(paste("Random selection - Crash before (True) and after intervention (False):", random_true_false), con = output_file)
writeLines(paste("Random selection - Crash before (False) and after intervention (True):", random_false_true), con = output_file)           

# Close the file
close(output_file)
