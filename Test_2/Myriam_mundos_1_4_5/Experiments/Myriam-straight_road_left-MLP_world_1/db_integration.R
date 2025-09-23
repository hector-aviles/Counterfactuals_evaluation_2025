# Integrate the entire database into a single file for OpenML or Kaggle.


library(stringr)
library(dplyr)
library(tidyverse)

# Get paths/names of all csv files in the directory (including files in any subdirectories)
myList <- csv_files <- system(paste("find", ".", "-type f -name '*.csv'", sep = " "), intern = TRUE)
myList

rm(df_integration)
df_integration = data.frame()
for (p in myList) {

  df <- read.table(p, header=T, sep = ",")
  condition <- str_detect(df$success, '^False$')  

  print(p)
  print(sprintf("Number of rows: %d", nrow(df)))
  print(sprintf("Number of successful decisions: %d",  sum(!condition))) 
  print(sprintf("Number of failures: %d", sum(condition)))   

     # append it to a new data frame 
     df_integration <- rbind(df_integration, df)
     
}

# Change the names of some columns
colnames(df_integration)[which(names(df_integration) == "free_north")] <- "free_N"

colnames(df_integration)[which(names(df_integration) == "free_north_east")] <- "free_NE"

colnames(df_integration)[which(names(df_integration) == "free_north_west")] <- "free_NW"

colnames(df_integration)[which(names(df_integration) == "free_west")] <- "free_W"

colnames(df_integration)[which(names(df_integration) == "free_south_west")] <- "free_SW"

colnames(df_integration)[which(names(df_integration) == "free_east")] <- "free_E"

colnames(df_integration)[which(names(df_integration) == "free_south_east")] <- "free_SE"

colnames(df_integration)[which(names(df_integration) == "car_1_pose.x")] <- "car1_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_1_pose.y")] <- "car1_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_2_pose.x")] <- "car2_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_2_pose.y")] <- "car2_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_3_pose.x")] <- "car3_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_3_pose.y")] <- "car3_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_4_pose.x")] <- "car4_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_4_pose.y")] <- "car4_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_5_pose.x")] <- "car5_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_5_pose.y")] <- "car5_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_6_pose.x")] <- "car6_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_6_pose.y")] <- "car6_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_7_pose.x")] <- "car7_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_7_pose.y")] <- "car7_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_8_pose.x")] <- "car8_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_8_pose.y")] <- "car8_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_9_pose.x")] <- "car9_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_9_pose.y")] <- "car9_pos.y"

colnames(df_integration)[which(names(df_integration) == "car_10_pose.x")] <- "car10_pos.x"
colnames(df_integration)[which(names(df_integration) == "car_10_pose.y")] <- "car10_pos.y"
  
colnames(df_integration)[which(names(df_integration) == "distance_to_north")] <- "front_vehicle_dist"  

colnames(df_integration)[which(names(df_integration) == "sdc_curr_pos.x")] <- "av_pos.x"  

colnames(df_integration)[which(names(df_integration) == "sdc_curr_pos.y")] <- "av_pos.y"  

colnames(df_integration)[which(names(df_integration) == "sdc_curr_pos.theta")] <- "av_pos.theta"  

df_integration <- subset(df_integration, select=-c(change_lane_finished, start_change_lane_on_left, start_change_lane_on_right, follow_enable, pass_finished))

#######################################################################
##
## Rename actions
##
########################################################################
# https://sparkbyexamples.com/r-programming/replace-empty-string-with-na-in-r-dataframe/#:~:text=Use%20df_integration%5Bdf_integration%3D%3D%E2%80%9D%5D,on%20all%20columns%20with%20NA.
df_integration$action[df_integration["action"] == ''] <- NA
df_integration$action[str_detect(df_integration$action, "Cruise")] <- "Cruise"
df_integration$action[str_detect(df_integration$action, "Keep")] <- "Keep"
df_integration$action[str_detect(df_integration$action, "Change lane 7")] <- "Change_to_left"
df_integration$action[str_detect(df_integration$action, "Change lane 15")] <- "Change_to_left"
df_integration$action[str_detect(df_integration$action, "Change lane 22")] <- "Change_to_right"
df_integration$action[str_detect(df_integration$action, "Change lane 24")] <- "Change_to_right"
df_integration$action[str_detect(df_integration$action, "Change lane 30")] <- "Change_to_right"
df_integration$action[str_detect(df_integration$action, "Change lane 32")] <- "Change_to_right"


############################################
##
## Remove incomplete cases
##
############################################
#orig <- nrow(df_integration)
#orig
#df_integration <- df_integration[complete.cases(df_integration), ]
#curr <- nrow(df_integration)
#curr
#removed <- orig - curr
#removed

####################################################################
##
## Set the value of free_N to the value of the variable 
## free_NW (resp. free_NE) when the car is on the left (resp. right)
## lane, and remove free_N.
##
####################################################################
# Create the condition
condition1 <- str_detect(df_integration$curr_lane, '^True$')

df_integration$free_NE <- ifelse(condition1, df_integration$free_N, df_integration$free_NE)
df_integration$free_NW <- ifelse(!condition1, df_integration$free_N, df_integration$free_NW)

df_integration <- subset(df_integration, select=-c(free_N))
  
# Define the search condition
condition <- str_detect(df_integration$success, '^False$')  

print(sprintf("Number of rows: %d", nrow(df_integration)))
print(sprintf("Number of successful decisions: %d",  sum(!condition))) 
print(sprintf("Number of failures: %d", sum(condition))) 
print(sprintf("Number of trials: %d", sum(df_integration$iteration == 1))) 

# Patch to introduce the missing action stop - modify the source code of the program 
df_integration <- df_integration %>%
mutate(action = ifelse(success == "False", "Stop", action))

write.csv(df_integration, file = "preprocessed.csv", row.names = FALSE)

names(df_integration)
dim(df_integration)

