import sys
import pandas as pd
import time
import statistics
import numpy as np
import os

def main(test_data, models_subdir):
    print("test_data received:", test_data, flush = True)

    # Load test data files
    input_csv = test_data + "/test_fold_{}.csv"
    input_csv_files = [input_csv.format(i) for i in range(1, 769)]
   
    # Load all models_subdir
    input_pl = models_subdir + "/cBN_{}.pl"
    input_pl_files = [input_pl.format(i) for i in range(1, 769)]
    
    time_list = [] 

    output_csv = models_subdir + "/twin_networks_results_{}.csv"    
    output_csv_files = [output_csv.format(i) for i in range(1, 769)]
    
    output = models_subdir + "/testing_numeralia.txt"
    with open(output, "w") as file:     
        file.write("\n")
        
    found_actions = models_subdir + "/found_actions.txt"
    with open(found_actions, "w") as file:     
        file.write("\n")        
    
    # Iterate through each fold
    for i in range(768):
        start_time = time.time()
        
        input_csv_file = input_csv_files[i]  
        print("input_csv_file:", input_csv_file, flush=True)
        input_pl_file = input_pl_files[i]
        print("input_pl_file:", input_pl_file, flush=True)
        output_csv_file = output_csv_files[i]
        print("output_csv_file:", output_csv_file, flush=True)
        
        # Construct the command for os.system()
        command = f"python3 run_WhatIf_V3.py {input_csv_file} {input_pl_file} {output_csv_file} {models_subdir} {found_actions}"
        print(f"Executing command: {command}", flush=True)
        
        exit_status = 0
        # Execute the command
        exit_status = os.system(command)
        if exit_status != 0:
            print(f"Error occurred during run_WhatIf_V2 for fold {i + 1}. Exit status: {exit_status}", flush=True)
            continue

        end_time = time.time()
        testing_time = end_time - start_time
        time_list.append(testing_time)
                
    average_time = statistics.mean(time_list)
    stdev_time = statistics.stdev(time_list)
   
    with open(output, "a") as file:
        string = "Avg. testing time: " + str(average_time) + "\n"
        file.write(string)
        string = "Stdev. testing time: " + str(stdev_time) + "\n"
        file.write(string)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test_cBNs.py <test_data> <models_subdir>")
        sys.exit(1)
    test_data = sys.argv[1]
    models_subdir = sys.argv[2]
    main(test_data, models_subdir)

