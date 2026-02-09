# Integrate the entire database into a single file for OpenML or Kaggle.

library(stringr)
library(dplyr)
library(tidyverse)

df_integration = data.frame()

# Get paths/names of all csv files in the directory (including files in any subdirectories)
myList <- c("integrated_DB_auto_humans_swerve.csv", "synthetic_DB.csv")
myList


for (p in myList) {

  df <- read.table(p, header=T, sep = ",")
  dim(df)
  # append it to a new data frame 
  df_integration <- rbind(df_integration, df)
     
}

dim(df_integration)
names(df_integration)

write.csv(df_integration, file = "complete_DB_discrete.csv", row.names = FALSE)



