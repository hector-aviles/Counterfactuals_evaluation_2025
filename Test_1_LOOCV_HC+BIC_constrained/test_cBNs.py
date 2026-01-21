#!/usr/bin/env python3
import sys
import os
import csv
import time
import statistics
from run_WhatIf_V4 import run_whatif

def process(rep_number, percentage):
    base_dir = os.getcwd()
    rep_dir = os.path.join(base_dir, f"rep_{rep_number}")
    perc_dir = os.path.join(rep_dir, str(percentage))
    
    test_data_dir = os.path.join(rep_dir, "test_data")
    models_subdir = os.path.join(perc_dir, "cBNs")
    
    os.makedirs(models_subdir, exist_ok=True)

    output_csv_path = os.path.join(models_subdir, "twin_networks_results.csv")
    numeralia_path = os.path.join(models_subdir, "testing_numeralia.txt")
    found_actions_path = os.path.join(models_subdir, "found_actions.txt")

    with open(numeralia_path, "w") as f:
        f.write(f"Testing numeralia summary for rep_{rep_number}/{percentage}\n\n")
    with open(found_actions_path, "w") as f:
        f.write("Fold,Action\n")

    num_folds = 768
    all_times = []

    with open(output_csv_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['action','curr_lane','free_E','free_NE','free_NW','free_SE','free_SW','free_W',
                         'orig_label_lc','latent_collision','iaction','probability','elapsed_time','group_id'])

        for i in range(1, num_folds + 1):
            input_csv = os.path.join(test_data_dir, f"test_fold_{i}.csv")
            input_pl = os.path.join(models_subdir, f"cBN_{i}.pl")
            if not os.path.exists(input_csv) or not os.path.exists(input_pl):
                print(f"[Warning] Missing input for fold {i}")
                continue

            start_fold = time.time()
            results, fold_times = run_whatif(input_csv, input_pl, models_subdir, found_actions_path, i)
            for row in results:
                writer.writerow(row)
            all_times.extend(fold_times)
            end_fold = time.time()
            print(f"Fold {i} done in {end_fold - start_fold:.2f}s, {len(results)} queries processed")

    # Write summary
    if all_times:
        avg_time = statistics.mean(all_times)
        std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0.0
    else:
        avg_time = std_time = 0.0

    with open(numeralia_path, "a") as f:
        f.write(f"Average testing time: {avg_time:.4f} s\n")
        f.write(f"Standard deviation:   {std_time:.4f} s\n")

    print(f"Average per query: {avg_time:.4f}s ± {std_time:.4f}s")

# Main
if __name__ == "__main__":
    reps = range(1, 6)
    percentages = ["01", "25", "50", "75", "90"]

    start_all = time.time()
    for rep in reps:
        for perc in percentages:
            process(rep, perc)
    end_all = time.time()
    print(f"All testing completed in {(end_all - start_all)/60:.2f} minutes")

