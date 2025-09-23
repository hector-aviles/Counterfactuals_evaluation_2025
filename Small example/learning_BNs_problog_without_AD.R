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
#install.packages("conflicted")
library(conflicted)  
#install.packages("tidyverse")
library(tidyverse)
conflict_prefer("filter", "dplyr")
conflict_prefer("lag", "dplyr")
# Required to use discretization function
#install.packages("arules")
library("arules")

#https://stackoverflow.com/questions/42256291/make-execution-stop-on-error-in-rstudio-interactive-r-session

# Define our error-handling pause function:
pause = function(){
    while (TRUE) {
        input <- readline(prompt="An error ocurred: Press CTRL-C key to continue (any subsequent code in the script was  ignored)")
    }    
}

# Set the 'error' option to execute our pause function:
options(error=pause)

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

args <- commandArgs(trailingOnly = TRUE)

datafile <- "../training_sample.csv"
output <- "../BN_problog.pl"

# Check if optional parameter is provided
if (length(args) > 0) {
    datafile <- args[1]
    output <- args[2]
} else {
    print("No parameters provided (it requires one csv file and the output file).")
}

print(paste("datafile: ", datafile))
print(paste("Output: ", output))  

# Read the database
df <- read.table(datafile, header=T, sep = ",")
df = subset(df, select = c(action, free_NE, free_W, latent_collision))
df <- subset(df, action == "cruise" | action == "keep" | action == "change_to_left" )

dim(df)
names(df)

df <- df[complete.cases(df), ]
dim(df)

bl_a <- data.frame(from = c("action", "action"), to = c("free_NE","free_W"))
bl_ne <- data.frame(from = c("free_NE"), to = c("free_W"))
bl_w <- data.frame(from = c("free_W"), to = c("free_NE"))
bl_lcol <- data.frame(from = c("latent_collision", "latent_collision", "latent_collision"), to = c("action",  "free_NE","free_W"))
bl <- rbind(bl_a, bl_ne, bl_w, bl_lcol)

# Convierte las variables a factores
df_factor <- as.data.frame(lapply(df, factor))
head(df_factor)

start_time <- Sys.time()
#network_structure <- pc.stable(df_factor, cluster = NULL, whitelist = NULL, blacklist = bl, test = "smc-mi", alpha = 0.05, B = NULL, max.sx = NULL, debug = TRUE, undirected = FALSE)
network_structure <-hc(df_factor,  whitelist = NULL, blacklist = bl, score = "bic", debug = FALSE, restart = 0, perturb = 1, max.iter = Inf, maxp = Inf, optimized = TRUE)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
print(paste0("Elapsed time (structural learning): ", elapsed_time))
flush.console()

model_filename <- paste(output, ".dot", sep = "", collapse = "")
write.dot(network_structure, file = model_filename)
# Plot the network
model_filename <- paste(output, ".png", sep = "", collapse = "")
png(model_filename)
plot(network_structure)
dev.off()

print(network_structure)

cmd <- paste("dot -Tps ", output, ".dot  -o ", output, ".ps", sep = "", collapse = "")
system(cmd, intern = TRUE)

start_time <- Sys.time()
# Learn parameters from data
bn_fit <- bn.fit(network_structure, data= df_factor, method="mle", replace.unidentifiable = TRUE)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
print(paste0("Elapsed time (parametric learning): ", elapsed_time))
flush.console()

# Display the learned parameters
print(bn_fit)

model_filename <- paste(output, ".net", sep = "", collapse = "")
write.net(model_filename, bn_fit)

output_file <- file(output, "w")
writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%", con = output_file)  
writeLines("% Probabilistic facts", con = output_file)  
writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n", con = output_file)  

# Extract facts
# RV is random variable
u_idx <- 0
for (rv in bn_fit) {
  rv_name <- rv$node
  rv_values <- dimnames(rv$prob)[[1]]   
  rv_numval = length(rv_values) 
       
  if (identical(rv$parents, character(0))){ 
      # the RV does not have parents
      print("Processing RV: ")
      flush.console()
      print(rv_name) # node's name

      # Convert the DAG data to a dataframe for a simpler extraction
      df2 <- as.data.frame(rv$prob)
      df2[] <- lapply(df2, as.character) # To simplify the following steps
      df2$Freq <- as.numeric(df2$Freq) # Return the last column to numeric data
      freq <- adjust_values(df2$Freq) # Normalize probabilities
      df2$Freq <- freq

      # Handle probabilistic facts based on number of values
      if (rv_numval == 2 && identical(tolower(rv_values[1]), 'false') && identical(tolower(rv_values[2]), 'true')) {
          # Binary true/false fact
          u_idx <- u_idx + 1
          fact <- paste(format(df2[2,]$Freq, nsmall=1, scientific=FALSE), "::u", u_idx, ".\n", 
                        rv_name, " :- u", u_idx, ".", sep = "", collapse = "")
          print(fact)
          writeLines(fact, con = output_file)
      } else {
          # Multi-valued or non-true/false binary fact
          for (i in seq(from = 1, to = rv_numval, by = 1)) {
              u_idx <- u_idx + 1
              # Write error term
              fact <- paste(format(df2[i,]$Freq, nsmall=1, scientific=FALSE), "::u", u_idx, ".", sep = "", collapse = "")
              # Write rule with mutual exclusivity
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
  } # identical(rv$parents, character(0))
} # for (rv in bn_fit)

writeLines("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%", con = output_file)  
writeLines("% Rules", con = output_file)  
writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n", con = output_file)

# Extract rules
for (rv in bn_fit) {
  rv_name <- rv$node
  rv_values <- dimnames(rv$prob)[[1]]   
  rv_numval = length(rv_values) 

  if (!identical(rv$parents, character(0))){ 
      # the RV has parents
      print("Processing RV: ")
      flush.console()
      print(rv_name) # node's name
      
      # Convert the DAG data to a dataframe to simplify data extraction
      df2 <- as.data.frame(rv$prob)
      df2[] <- lapply(df2, as.character) # To simplify the following steps
      df2$Freq <- as.numeric(df2$Freq) # Return the last column as numeric data
  
      # Create rules for each combination of parent values
      for (i in seq(from = 1, to = nrow(df2), by = rv_numval)) {
         lower_lim <- i
         upper_lim <- i + rv_numval - 1
         freq <- df2$Freq[lower_lim:upper_lim]
         df2$Freq[lower_lim:upper_lim] <- adjust_values(freq) 

         if (rv_numval == 2 && identical(tolower(rv_values[1]), 'false') && identical(tolower(rv_values[2]), 'true')) {
             # Binary true/false rule
             u_idx <- u_idx + 1
             head <- paste(format(df2[i + 1,]$Freq, nsmall=1, scientific=FALSE), "::u", u_idx, ".", sep = "", collapse = "")
             body <- paste(rv_name, " :- ", sep = "", collapse = "")
             for (k in seq(from = 2, to = ncol(df2) - 1, by = 1)) {
                 col_values <- unique(as.character(unlist(df2[, k])))
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
                 if (k < ncol(df2) - 1) {
                     body <- paste(body, ", ", sep = "", collapse = "")
                 }
             }
             body <- paste(body, ", u", u_idx, ".", sep = "", collapse = "")
             rule <- paste(head, "\n", body, "\n", sep = "", collapse = "")
             print(rule)
             writeLines(rule, con = output_file)
         } else {
             # Multi-valued or non-true/false binary rule
             for (j in seq(from = i, to = i + rv_numval - 1, by = 1)) {
                 u_idx <- u_idx + 1
                 head <- paste(format(df2[j,]$Freq, nsmall=1, scientific=FALSE), "::u", u_idx, ".", sep = "", collapse = "")
                 body <- paste(rv_name, "(", df2[j,1], ") :- ", sep = "", collapse = "")
                 # Add mutual exclusivity constraints with commas
                 for (m in seq(from = 1, to = rv_numval, by = 1)) {
                     if (m != (j - i + 1)) {
                         if (body == paste(rv_name, "(", df2[j,1], ") :- ", sep = "", collapse = "")) {
                             # First negated atom, no comma needed before
                             body <- paste(body, "\\+ ", rv_name, "(", rv_values[m], ")", sep = "", collapse = "")
                         } else {
                             # Subsequent negated atoms, add comma before
                             body <- paste(body, ", \\+ ", rv_name, "(", rv_values[m], ")", sep = "", collapse = "")
                         }
                     }
                 }
                 for (k in seq(from = 2, to = ncol(df2) - 1, by = 1)) {
                     col_values <- unique(as.character(unlist(df2[, k])))
                     if (length(col_values) > 2) {
                         body <- paste(body, ", ", names(df2)[k], "(", df2[j, k], ")", sep = "", collapse = "")
                     } else {
                         if (identical(tolower(col_values[1]), 'false') && identical(tolower(col_values[2]), 'true')) {
                             if (identical(tolower(df2[j, k]), 'false')) {
                                 body <- paste(body, ", \\+ ", names(df2)[k], sep = "", collapse = "")
                             } else {
                                 body <- paste(body, ", ", names(df2)[k], sep = "", collapse = "")
                             }
                         } else {
                             body <- paste(body, ", ", names(df2)[k], "(", df2[j, k], ")", sep = "", collapse = "")
                         }
                     }
                 }
                 body <- paste(body, ", u", u_idx, ".", sep = "", collapse = "")
                 rule <- paste(head, "\n", body, "\n", sep = "", collapse = "")
                 print(rule)
                 writeLines(rule, con = output_file)
             }
         }
      } # for (i in seq
  } # !identical(rv$parents, character(0))
} # for (rv in bn_fit)

close(output_file)
options(error=NULL)
