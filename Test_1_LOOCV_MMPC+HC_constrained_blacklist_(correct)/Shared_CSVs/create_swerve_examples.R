# Load required library
library(tidyverse)

# Define all binary variables (True/False)
binary_vars <- c("curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W")

# Generate all possible combinations (2^7 = 128 permutations)
all_combinations <- expand.grid(
  lapply(binary_vars, function(x) c("True", "False")),
  stringsAsFactors = FALSE
)

# Name the columns
colnames(all_combinations) <- binary_vars

# Add "swerve_left" and "swerve_right" actions (duplicate rows)
swerve_left <- all_combinations %>%
  mutate(action = "swerve_left")

swerve_right <- all_combinations %>%
  mutate(action = "swerve_right")

# Combine into one dataset
synthetic_data <- bind_rows(swerve_left, swerve_right) %>%
  mutate(latent_collision = "False") %>%
  select(action, everything())  # Reorder columns so 'action' comes first

# Replicate each row 500 times (256 x 500 = 128,000 rows)
synthetic_data_expanded <- synthetic_data %>%
  slice(rep(1:n(), each = 500))

# Save to CSV with quoting (base R)
#write.csv(synthetic_data, "unique_swerve_examples.csv", 
#          row.names = FALSE, 
#          quote = TRUE)  # Forces quoting of all strings

# Save to CSV with quoting (base R)
#write.csv(synthetic_data_expanded, "synthetic_swerve.csv", 
#          row.names = FALSE, 
#          quote = TRUE)  # Forces quoting of all strings

# --- Step 2: Load Existing Dataset ---
# Read the complete_DB_discrete.csv file
#existing_data <- read_csv("complete_DB_discrete.csv")
existing_data <- read.table("complete_DB_discrete.csv", header=T, sep = ",")

# Ensure column names match (adjust if needed)
colnames(existing_data) <- colnames(synthetic_data_expanded)

# --- Step 3: Merge Datasets ---
# Combine the existing data with synthetic swerve data
complete_DB <- bind_rows(existing_data, synthetic_data_expanded)

# --- Step 4: Save Merged Dataset ---
# Save to CSV with quoting (base R)
write.csv(complete_DB, "complete_DB.csv", 
          row.names = FALSE, 
          quote = TRUE)  # Ensures all strings are quoted

# Print confirmation
cat("Merged dataset saved to 'complete_DB.csv'.\n")
cat("Original rows:", nrow(existing_data), "\n")
cat("Synthetic rows added:", nrow(synthetic_data_expanded), "\n")
cat("Total rows in output:", nrow(complete_DB), "\n")

# Print confirmation
#cat("Generated", nrow(synthetic_data), "rows and saved to 'synthetic_swerve.csv'.\n")
