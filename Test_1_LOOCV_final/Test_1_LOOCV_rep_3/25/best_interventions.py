import sys
import pandas as pd
import time
import statistics
import numpy as np
import os

def main(input_subdir, reference_filename):
    print("input_subdir received:", input_subdir, flush=True)
    print("reference_filename received:", reference_filename, flush=True)

    # Load test data files
    input_csv = input_subdir + "/twin_networks_results.csv"
    bi_csv = input_subdir + "/best_interventions.csv"    
    ds_csv = input_subdir + "/data_sorted.csv"    
    ct_csv = input_subdir + "/contingency_table.csv"    
        
    # Iterate through each fold
    for i in range(1):
                
        # Construct the command for os.system()
        command = f"Rscript best_interventions.R {input_csv}  {bi_csv} {reference_filename} {ds_csv} {ct_csv} "
        print(f"Executing command: {command}", flush=True)
        
        # Execute the command
        exit_status = os.system(command)
        if exit_status != 0:
            print(f"Error occurred during best_interventions.R for fold {i + 1}. Exit status: {exit_status}", flush=True)
            continue
        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 best_interventions.py <input_subdir>  no_crash_filename")
        sys.exit(1)
    input_subdir = sys.argv[1]
    reference_filename = sys.argv[2]    
    main(input_subdir, reference_filename)

