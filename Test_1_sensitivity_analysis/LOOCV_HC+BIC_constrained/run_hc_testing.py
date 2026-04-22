#!/usr/bin/env python3
# =============================================================================
# run_hc_testing.py
#
# Step 2: counterfactual probability evaluation for one HC+BIC sensitivity
# combination. Calls run_WhatIf_V3.py once per fold (keeping CounterfactualProgram
# alive across all 6 iaction rows within each fold).
#
# Usage:
#   python3 run_hc_testing.py <combo_dir> [--folds 1-78]
#
# Arguments:
#   combo_dir   : path to a combination output directory, e.g.
#                 ./hc_bic_restart0_perturb1_maxpInf_fracp25
#   --folds     : fold range to process, e.g. "1-78" (default) or "5-10"
#                 for resuming after interruption
#
# Output per fold:
#   <combo_dir>/fold_<N>/twin_networks_results.csv  (same format as original)
#   <combo_dir>/fold_<N>/found_actions.txt
#   <combo_dir>/fold_<N>/elapsed_time.csv
#
# Aggregate output (appended, safe to resume):
#   <combo_dir>/twin_networks_results_all.csv       (all folds concatenated)
#   <combo_dir>/testing_numeralia.txt
# =============================================================================

import os
import sys
import csv
import time
import statistics

WHATIF_SCRIPT = "run_WhatIf_V3.py"
CSV_HEADER    = [
    'action', 'curr_lane', 'free_E', 'free_NE', 'free_NW',
    'free_SE', 'free_SW', 'free_W', 'orig_label_lc',
    'latent_collision', 'iaction', 'probability', 'elapsed_time', 'group_id'
]


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 run_hc_testing.py <combo_dir> [--folds START-END]")
        sys.exit(1)

    combo_dir  = args[0]
    fold_start = 1
    fold_end   = 78   # default: all folds

    i = 1
    while i < len(args):
        if args[i] == "--folds" and i + 1 < len(args):
            parts = args[i+1].split("-")
            if len(parts) == 2:
                fold_start = int(parts[0])
                fold_end   = int(parts[1])
            else:
                fold_start = fold_end = int(parts[0])
            i += 2
        else:
            i += 1

    return combo_dir, fold_start, fold_end


# =============================================================================
# Helpers
# =============================================================================
def fold_dir_path(combo_dir, fold_idx):
    return os.path.join(combo_dir, f"fold_{fold_idx}")


def count_existing_folds(combo_dir, fold_start, fold_end):
    """Return list of fold indices whose twin_networks_results.csv already exists."""
    done = []
    for fi in range(fold_start, fold_end + 1):
        result = os.path.join(fold_dir_path(combo_dir, fi), "twin_networks_results.csv")
        if os.path.exists(result) and os.path.getsize(result) > 0:
            done.append(fi)
    return done


def write_csv_header(path):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_fold_results_to_aggregate(fold_result_csv, aggregate_csv):
    """Append rows from a fold result CSV (no header) into the aggregate file."""
    if not os.path.exists(fold_result_csv):
        return 0
    rows_written = 0
    with open(fold_result_csv, 'r', newline='') as src, \
         open(aggregate_csv, 'a', newline='') as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        for j, row in enumerate(reader):
            if j == 0 and row == CSV_HEADER:
                continue  # skip header row from fold file
            writer.writerow(row)
            rows_written += 1
    return rows_written


# =============================================================================
# Main
# =============================================================================
def main():
    combo_dir, fold_start, fold_end = parse_args()

    if not os.path.isdir(combo_dir):
        print(f"[Error] combo_dir not found: {combo_dir}")
        sys.exit(1)

    if not os.path.exists(WHATIF_SCRIPT):
        print(f"[Error] {WHATIF_SCRIPT} not found in {os.getcwd()}")
        sys.exit(1)

    combo_name       = os.path.basename(os.path.abspath(combo_dir))
    aggregate_csv    = os.path.join(combo_dir, "twin_networks_results_all.csv")
    numeralia_path   = os.path.join(combo_dir, "testing_numeralia.txt")

    print(f"\n{'='*60}")
    print(f"  HC sensitivity testing: {combo_name}")
    print(f"  Folds: {fold_start} to {fold_end}")
    print(f"  combo_dir: {combo_dir}")
    print(f"{'='*60}\n")

    # Initialize aggregate CSV with header if starting fresh
    already_done = count_existing_folds(combo_dir, fold_start, fold_end)
    if already_done:
        print(f"[Resume] Found {len(already_done)} completed folds: {already_done}")
        print(f"  Skipping completed folds — delete fold_<N>/twin_networks_results.csv to rerun.\n")

    if not os.path.exists(aggregate_csv):
        write_csv_header(aggregate_csv)
        print(f"Aggregate CSV initialized: {aggregate_csv}")

    # Append header to numeralia if new run
    with open(numeralia_path, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Testing run: {combo_name}\n")
        f.write(f"Folds: {fold_start}-{fold_end}\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # Fold loop
    # ------------------------------------------------------------------
    fold_times   = []
    skipped      = []
    failed       = []

    for fi in range(fold_start, fold_end + 1):
        fdir = fold_dir_path(combo_dir, fi)

        # Paths
        model_pl      = os.path.join(fdir, "cBN.pl")
        test_csv      = os.path.join(fdir, "test_data.csv")
        result_csv    = os.path.join(fdir, "twin_networks_results.csv")
        actions_found = os.path.join(fdir, "found_actions.txt")

        # Skip if already completed
        if fi in already_done:
            print(f"[Skip] Fold {fi} already done.", flush=True)
            skipped.append(fi)
            continue

        # Check required inputs
        if not os.path.exists(model_pl):
            print(f"[Warning] Missing model: {model_pl} — skipping fold {fi}", flush=True)
            failed.append(fi)
            continue
        if not os.path.exists(test_csv):
            print(f"[Warning] Missing test data: {test_csv} — skipping fold {fi}", flush=True)
            failed.append(fi)
            continue

        print(f"\n{'─'*50}")
        print(f"  Fold {fi}/{fold_end} | {combo_name}", flush=True)
        print(f"{'─'*50}")

        # Initialize per-fold result CSV with header
        write_csv_header(result_csv)

        # Initialize found_actions for this fold
        with open(actions_found, 'w') as f:
            f.write(f"Fold {fi}\n")

        # ------------------------------------------------------------------
        # Call run_WhatIf_V3.py — CounterfactualProgram created once per fold
        # inside that script, reused across all 6 iaction rows
        # ------------------------------------------------------------------
        t0 = time.time()
        cmd = (
            f"python3 {WHATIF_SCRIPT} "
            f"{test_csv} {model_pl} {result_csv} "
            f"{fdir} {actions_found} {fi}"
        )
        print(f"  Executing: {cmd}", flush=True)
        exit_status = os.system(cmd)
        elapsed = time.time() - t0

        if exit_status != 0:
            print(f"[Error] run_WhatIf_V3.py failed for fold {fi} (exit {exit_status})", flush=True)
            failed.append(fi)
            # Remove partial result to allow clean rerun
            if os.path.exists(result_csv):
                os.remove(result_csv)
            continue

        fold_times.append(elapsed)
        print(f"  Fold {fi} completed in {elapsed:.2f}s", flush=True)

        # Append this fold's results to aggregate CSV
        n_rows = append_fold_results_to_aggregate(result_csv, aggregate_csv)
        print(f"  Appended {n_rows} rows to aggregate CSV", flush=True)

        # Per-fold numeralia entry
        with open(numeralia_path, 'a') as f:
            f.write(f"Fold {fi}: {elapsed:.2f}s | rows={n_rows}\n")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    n_done    = len(fold_times)
    n_skipped = len(skipped)
    n_failed  = len(failed)

    avg_t = statistics.mean(fold_times)   if fold_times else 0.0
    std_t = statistics.stdev(fold_times)  if len(fold_times) > 1 else 0.0

    summary = (
        f"\n{'='*50}\n"
        f"SUMMARY: {combo_name}\n"
        f"Folds processed : {n_done}\n"
        f"Folds skipped   : {n_skipped} (already done)\n"
        f"Folds failed    : {n_failed}"
        + (f" {failed}" if failed else "") + "\n"
        f"Avg time/fold   : {avg_t:.2f} +/- {std_t:.2f} s\n"
        f"Aggregate CSV   : {aggregate_csv}\n"
        f"Finished        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*50}\n"
    )

    print(summary, flush=True)
    with open(numeralia_path, 'a') as f:
        f.write(summary)

    if failed:
        print(f"[Warning] {n_failed} fold(s) failed: {failed}")
        print(f"  Rerun with: python3 run_hc_testing.py {combo_dir} --folds <N>-<N>")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
