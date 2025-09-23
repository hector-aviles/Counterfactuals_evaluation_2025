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

  # bn_fit$free_NE$node
  rv_name <- rv$node

  # Values taken by rv shown in the table
  rv_values <- dimnames(rv$prob)[[1]]   
  # Check whether the current RV has parents (a rule must be created) or not (a random fact must be written)
  rv_numval = length(rv_values) 
       
  if (identical(rv$parents, character(0))){ 
  
      # the RV does not have parents

      print("Processing RV: ")
      flush.console()
      print(rv_name) # node's name

      # index of the auxiliary probabilistic facts u* (required by WhatIf)
      u_idx <- u_idx + 1

      # Convert the DAG data to a dataframe for a simpler extraction
      df2 <- as.data.frame(rv$prob)
      # The order of the next instructions is important
      # Replace NA values with 0.0 
      df2[is.na(df2)] <- 0.0
      df2[] <- lapply(df2, as.character) # To simplify the following steps
      df2$Freq <- as.numeric(df2$Freq) # Return the last column to numeric data
    
  # Annotated disjunctions (not test yet with WhatIf)
      fact <- ""
      lower_lim <- 1
      upper_lim <- rv_numval - 1
      freq <- df2$Freq[lower_lim:upper_lim]
      df2$Freq[lower_lim:upper_lim] <- adjust_values(freq)       
      for (i in seq(from = 1, to = rv_numval, by = 1)){
          # Normalize values to 1, eliminates NA and set low values lesser than 0.0001 to 0.0 (as required by problog)

          if ((rv_numval == 2) & identical(tolower(rv_values[1]), 'false') & identical(tolower(rv_values[2]), 'true')){

            fact <- paste(format(df2[i+1,]$Freq, nsmall=1, scientific=FALSE), "::u", u_idx, ".\n", rv_name, " :- ", "u", u_idx, sep = "", collapse = "")

            break 

          }   

          fact <- paste(fact, df2[i, ]$Freq, "::", rv_name, 
                      "(", df2[i,1], ")", sep = "", collapse = "")
          # Add ';'
          if (i < rv_numval){
             fact <- paste(fact, "; ", sep = "", collapse = "")
          } 
 
      } # (i in seq(from = 1, to = rv_numval, by = 1))
         
      fact <- paste(fact, ".", sep = "", collapse = "")
      print(fact)
      writeLines(fact, con = output_file)  

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

  # Check whether the current RV has parents (a rule must be created) or not (a random fact should have been written)       
  if (!identical(rv$parents, character(0))){ 
      # the RV has parents
      print("Processing RV: ")
      flush.console()
      print(rv_name) # node's name
      
      uflag <- FALSE
      
      # Convert the DAG data to a dataframe to simplify data extraction
      df2 <- as.data.frame(rv$prob)
      df2
      # The order of the next instructions is important
      # Replace NA values with 0.0 
      df2[] <- lapply(df2, as.character) # To simplify the following steps
      df2$Freq <- as.numeric(df2$Freq) # Return the last column as numeric data
  
      # Create the rules one by one ("rv_numval" rows)
      # i scans each row to construct the rules of the RV
      for (i in seq(from = 1, to = nrow(df2), by = rv_numval)){
         # create the head of the rule first      
         head <- ""
          
         # Normalize values to 1, eliminates NA and set low values lesser than 0.0001 to 0.0 (as required by problog)         
         lower_lim <- i
         upper_lim <- i + rv_numval - 1
         freq <- df2$Freq[lower_lim:upper_lim]
         df2$Freq[lower_lim:upper_lim] <- adjust_values(freq) 

         # j indexes the values of the RV 
         for (j in seq(from = i, to = i + rv_numval - 1, by = 1)){
         
            # Set the head of the rule  
            if (identical(tolower(rv_values[1]), 'false') &
                identical(tolower(rv_values[2]), 'true')){
                # Boolean RV in the head (j + 1 because we need the probability of the RV being 'true')
                head <- paste(head, format(df2[j + 1, ]$Freq, nsmall=1,  scientific=FALSE), "::",  rv_name,  sep = "", collapse = "")
                               
                uflag <- TRUE                                             
                # Go to the next row                 
                break    
            }
                       
            head <- paste(head, format(df2[j, ]$Freq, nsmall=1, scientific=FALSE), "::", rv_name, "(", df2[j,1], ")", sep = "", collapse = "")
                           	
            # Add ';' for annotated disjunctions
            if (j < i + rv_numval - 1){
               head <- paste(head, "; ", sep = "", collapse = "")
            }  

         } # for (j in seq(from = i,

         # if the frequency of the values of the head is equal to zero,
         # do not create the rule 
         #if (count_zeros < rv_numval){
         # create the body
         body <- ""
         # Scan every column of the parent RVs
         for (k in seq(from = 2, to = ncol(df2) - 1, by = 1)){
            # Check if the k-th parent RV uses annotated disjunctions (I know the value of the current RV but not the values of its parents)
            col_values <- unique(as.character(unlist(df2[, k])))
            flush.console()
                  
            if (length(col_values) > 2){ # RV with more than 2 values 
              body <- paste(body, names(df2)[k], "(", df2[i, k], ")", 
                            sep = "", collapse = "")      
            } else {
               # The parent RV is Boolean
               if (identical(tolower(col_values[1]), 'false') &
                    identical(tolower(col_values[2]), 'true')){  
                   if (identical(tolower(df2[i, k]), 'false')){
                      body <- paste(body, "\\+ ", 
                              names(df2)[k], sep = "", collapse = "")
                   } else {
                      body <- paste(body,  
                              names(df2)[k], sep = "", collapse = "")
                   }
                            
               } else { # if (identical(tolower(col_values[1])
                     body <- paste(body, names(df2)[k], "(", 
                         df2[i, k], ")", sep = "", collapse = "")
               }
                  
            } 
                  
            if (k == ncol(df2) - 1){
               if (uflag){
                  # For binary variables add an u* variable 
                  # at the end of the body
                  body <- paste(body, ".", sep = "", collapse = "") 
               } else {
                  body <- paste(body, ".", sep = "", collapse = "")                
               }
                  
            } else {
               body <- paste(body, ", ", sep = "", collapse = "") 
            }
               
         } # (k in seq(from = 2, to = ncol(df) - 1, by = 1))  

         rule <- paste(head, ":-", body, "\n\n")
         print(rule)
         writeLines(rule, con = output_file)
               
         #} #  (count_zeros < rv_numval)   
                          
      } # for (i in seq(from = 1   

  } # !identical(rv$parents, character(0))

} # for (rv in bn_fit)

close(output_file)

options(error=NULL)


