library(dplyr)

base_path <- "./cBNs"
twin_networks_files <- paste0(base_path, "/twin_networks_results_", 1:768, ".csv")
output_best_int <- paste0(base_path, "/twin_networks_results.csv")
missing_files_log <- paste0(base_path, "/missing_files.txt")

combine_csv <- function(input_files, output_file, missing_log) {
  # Initialize vectors for found and missing files
  found_files <- c()
  missing_files <- c()
  file_row_counts <- list()
  
  # Debug: Confirm total number of files being checked
  message(sprintf("Checking %d input files...", length(input_files)))
  
  # Check each file
  for (file in input_files) {
    if (file.exists(file)) {
      found_files <- c(found_files, file)
      data <- read.csv(file, stringsAsFactors = FALSE)
      file_row_counts[[basename(file)]] <- nrow(data)
    } else {
      missing_files <- c(missing_files, file)
    }
  }
  
  # Report found files and row counts
  if (length(found_files) > 0) {
    message(sprintf("\nFound %d input files.", length(found_files)))
    message("\nRow counts for input files:")
    for (file in names(file_row_counts)) {
      if (file_row_counts[[file]] < 6) {
        message(sprintf("ERROR!: %s: %d rows", file, file_row_counts[[file]]))
      } else {
        message(sprintf("  %s: %d rows", file, file_row_counts[[file]]))
      }
    }
  } else {
    stop("No input files found to combine. Check the directory and file names.")
  }
  
  # Report and log missing files
  message(sprintf("\nComplete List of Missing Files (%d total):", length(missing_files)))
  if (length(missing_files) == 0) {
    message("  None")
    writeLines("No missing files.", con = missing_log)
  } else {
    for (file in missing_files) {
      message(sprintf("  %s", basename(file)))
    }
    writeLines(c(sprintf("Complete List of Missing Files (%d total):", length(missing_files)),
                 basename(missing_files)), con = missing_log)
  }
  message(sprintf("\nMissing files list saved to %s", missing_log))
  
  # Combine found files
  all_data <- lapply(found_files, function(file) {
    read.csv(file, stringsAsFactors = FALSE)
  })
  
  if (length(all_data) > 0) {
    combined_data <- do.call(rbind, all_data)
    message(sprintf("\nOutput file will contain %d rows.", nrow(combined_data)))
    write.csv(combined_data, output_file, row.names = FALSE)
    if (file.exists(output_file)) {
      output_data <- read.csv(output_file, stringsAsFactors = FALSE)
      message(sprintf("Combined data successfully saved to %s with %d rows.", output_file, nrow(output_data)))
    } else {
      stop("Failed to write output file: ", output_file)
    }
  } else {
    stop("No data could be read from the input files.")
  }
}

tryCatch({
  combine_csv(twin_networks_files, output_best_int, missing_files_log)
}, error = function(e) {
  stop("Failed to combine best_interventions files: ", e$message)
})
