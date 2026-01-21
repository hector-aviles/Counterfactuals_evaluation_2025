#!/usr/bin/env python3
import sys
import pandas as pd
import time
import statistics
import os
import csv

# ======================================
# Function to process a single rep/percentage pair
# ======================================
def process(rep_number, percentage):
    base_dir = os.getcwd()
    rep_dir = os.path.join(base_dir, f"rep_{rep_number}")
    perc_dir = os.path.join(rep_dir, str(percentage))
    
    test_data_dir = os.path.join(rep_dir, "test_data")
    models_subdir = os.path.join(perc_dir, "cBNs")
    
    print(f"\n=== Running tests for rep_{rep_number}/{percentage} ===")
    print(f"Test data directory: {test_data_dir}")
    print(f"Models directory: {models_subdir}", flush=True)

    # Expected files
    num_folds = 768
    input_csv_pattern = os.path.join(test_data_dir, "test_fold_{}.csv")
    input_pl_pattern = os.path.join(models_subdir, "cBN_{}.pl")

    # Output files
    output_csv_path = os.path.join(models_subdir, "twin_networks_results.csv")
    numeralia_path = os.path.join(models_subdir, "testing_numeralia.txt")
    found_actions_path = os.path.join(models_subdir, "found_actions.txt")
     
    # Initialize outputs
    os.makedirs(models_subdir, exist_ok=True)
    with open(numeralia_path, "w") as f:
        f.write(f"Testing numeralia summary for rep_{rep_number}/{percentage}\n\n")
    with open(found_actions_path, "w") as f:
        f.write("Fold,Action\n")


    time_list = []

    # ======================================
    # Iterate through folds
    # ======================================
    for i in range(1, num_folds + 1):
        start_time = time.time()
        
        if i == 1:
         # If results file exists, remove to avoid appending old data
         if os.path.exists(output_csv_path):
           os.remove(output_csv_path)
         # Write header to output
         with open(output_csv_path, 'w', newline='') as outfile:
           writer = csv.writer(outfile, delimiter=',')
           writer.writerow(['action','curr_lane','free_E','free_NE',  'free_NW','free_SE','free_SW','free_W', 'orig_label_lc', 'latent_collision', 'iaction', 'probability', 'elapsed_time', 'group_id'])           
               
        input_csv = input_csv_pattern.format(i)
        input_pl = input_pl_pattern.format(i)

        if not os.path.exists(input_csv) or not os.path.exists(input_pl):
            print(f"[Warning] Missing input for fold {i}: {input_csv} or {input_pl}", flush=True)
            continue

        print(f"\n--- Fold {i}/{num_folds} ---", flush=True)
        #print(f"Input test file: {input_csv}")
        #print(f"Input model file: {input_pl}")

        command = f"python3 run_WhatIf_V3.py {input_csv} {input_pl} {output_csv_path} {models_subdir} {found_actions_path} {i}"
        #print(f"Executing: {command}", flush=True)
        
        exit_status = os.system(command)
        if exit_status != 0:
            print(f"[Error] run_WhatIf_V3.py failed for fold {i} (exit {exit_status})", flush=True)
            continue
         
        end_time = time.time()
        elapsed = end_time - start_time
        time_list.append(elapsed)
        print(f"Fold {i} done in {elapsed:.2f} s", flush=True)

    # ======================================
    # Summary statistics
    # ======================================
    if len(time_list) > 1:
        avg_time = statistics.mean(time_list)
        std_time = statistics.stdev(time_list)
    elif len(time_list) == 1:
        avg_time = time_list[0]
        std_time = 0.0
    else:
        avg_time = std_time = 0.0

    with open(numeralia_path, "a") as f:
        f.write(f"Average testing time: {avg_time:.4f} s\n")
        f.write(f"Standard deviation:   {std_time:.4f} s\n")

    print(f"Average time per fold: {avg_time:.2f} s ± {std_time:.2f} s", flush=True)
    print(f"Numeralia file saved to: {numeralia_path}\n", flush=True)


# ======================================
# MAIN LOOP OVER REPETITIONS AND PERCENTAGES
# ======================================
if __name__ == "__main__":
    reps = range(1, 6)
    percentages = ["01", "25", "50", "75", "90"]

    # Optional CLI arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "reps":
            reps = range(1, int(sys.argv[2]) + 1)
        elif sys.argv[1] == "percentages":
            percentages = sys.argv[2].split(",")
        elif sys.argv[1] == "both":
            reps = range(1, int(sys.argv[2]) + 1)
            percentages = sys.argv[3].split(",")

    start_all = time.time()
    for rep in reps:
        for perc in percentages:
            process(rep, perc)
    end_all = time.time()

    print(f"\n=== All testing completed in {(end_all - start_all)/60:.2f} minutes ===\n")

