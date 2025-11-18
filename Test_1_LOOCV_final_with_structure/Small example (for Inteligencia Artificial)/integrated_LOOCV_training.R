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

# Set the decimal rounding
dec_round <- 7

args <- commandArgs(trailingOnly = TRUE)
if (length(args) !=2) stop("Usage: Rscript script.R <input> <output>")
input <- args[1]
output_pl <- args[2]

input <- "training_data_01_fold_1.csv"
output_pl <- "cBN.pl"

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

# -----------------------
# Input validation: shared CSVs exist
# -----------------------
input_files <- file.path("./", c(input))
for (f in input_files) {
  if (!file.exists(f)) stop("Input file missing: ", f)
}

# -----------------------
# Read core datasets once (we'll copy per rep to ensure reproducibility)
# -----------------------
# --- Read datasets ---
dt <- read.table(input, header=T, sep = ",")
dt <- subset(dt, select = c(action, free_NE, free_W, latent_collision))
dt <- subset(dt, action == "cruise" | action == "keep" | action == "change_to_left") 
#dt <- dt[complete.cases(dt)]
dt_factor <- as.data.frame(lapply(dt, factor))
head(dt_factor)

# --- Blacklist construction
bl_a <- data.frame(from = rep("action", 2), to = c("free_NE", "free_W"))

bl_lcol <- data.frame(from = rep("latent_collision", 3), to = c("action",  "free_NE",  "free_W"))

bl_ne <- data.frame(from = rep("free_NE", 1), to = c("free_W"))

bl_w <- data.frame(from = rep("free_W", 1), to = c("free_NE"))

bl <- rbind(bl_a, bl_lcol, bl_ne, bl_w)

# --- Structural learning (hc)
network_structure <- hc(dt_factor, whitelist = NULL, blacklist = bl, score = "bic", debug = FALSE, restart = 0, perturb = 1, max.iter = Inf, maxp = Inf, optimized = TRUE)

# --- Save network plot and dot
output_base <- file.path("./", "cBN")
model_dot <- paste0(output_base, ".dot")
write.dot(network_structure, file = model_dot)
# Also create PS via dot if dot is available on system
try({
  cmd <- paste("dot -Tps", shQuote(model_dot), "-o", shQuote(paste0(output_base, ".ps")))
  system(cmd, intern = TRUE)
}, silent = TRUE)

# --- Parametric learning (bn.fit)
bn_fit <- bn.fit(network_structure, data = dt_factor, method = "mle", replace.unidentifiable = TRUE)

# --- Write parameters to .net file (bnlearn write.net)
model_net_fn <- paste0(output_base, ".net")
try({
  write.net(model_net_fn, bn_fit)
}, silent = TRUE)

# --- Write the .pl probabilistic facts and rules (original logic)
# Write the probabilistic logic program
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

close(output_file)




