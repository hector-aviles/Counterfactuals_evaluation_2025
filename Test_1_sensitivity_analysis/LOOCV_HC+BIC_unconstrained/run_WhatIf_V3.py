#!/usr/bin/env python3
# =============================================================================
# run_WhatIf_V3.py
#
# Runs counterfactual queries for a single fold using a fitted PLTN.
# Called once per fold by run_hc_testing.py.
# The CounterfactualProgram object is created once and reused for all 6 iactions.
#
# Usage:
#   python3 run_WhatIf_V3.py <input.csv> <input.pl> <output.csv> \
#                             <models_subdir> <output_actions_found> <fold_index>
# =============================================================================

import sys
import re
import csv
import time

from counterfactuals.counterfactualprogram import CounterfactualProgram
import aspmc.config as config
from aspmc.main import logger as aspmc_logger


def main(input_csv, input_cbn, output_csv, models_subdir, output_actions_found, fold):

    print(f"Input CSV  : {input_csv}",   flush=True)
    print(f"Input PL   : {input_cbn}",   flush=True)
    print(f"Output CSV : {output_csv}",  flush=True)
    print(f"Models dir : {models_subdir}", flush=True)

    # ------------------------------------------------------------------
    # Detect which actions are present in the .pl file
    # ------------------------------------------------------------------
    actions_list = ["change_to_left", "change_to_right", "cruise",
                    "keep", "swerve_left", "swerve_right"]
    actions = []
    try:
        with open(input_cbn, 'r') as f:
            content = f.read()
        for action in actions_list:
            if re.search(rf"{action}", content):
                actions.append(action)
        actions = list(dict.fromkeys(actions))  # deduplicate, preserve order
        with open(output_actions_found, "a") as f:
            f.write(str(actions) + "\n")
    except FileNotFoundError:
        print(f"[Warning] Could not read {input_cbn}, using full actions list", flush=True)
        actions = actions_list[:]

    # Build truth tuples for interventions
    # iaction_truth[i] = tuple where position i is 'True', rest 'False'
    iaction_truth = []
    for i in range(len(actions)):
        tv = ['False'] * len(actions)
        tv[i] = 'True'
        iaction_truth.append(tuple(tv))

    print(f"Detected actions : {actions}",       flush=True)
    print(f"Truth tuples     : {iaction_truth}", flush=True)

    # Elapsed time log
    elapsed_time_path = f"{models_subdir}/elapsed_time.csv"
    with open(elapsed_time_path, "w") as f:
        f.write(f"%%%% {output_csv}\n")

    # ------------------------------------------------------------------
    # Load the BN once — reused for all 6 iaction queries in this fold
    # ------------------------------------------------------------------
    program = CounterfactualProgram("", [input_cbn])

    # aspmc verbosity
    config.config["knowledge_compiler"] = "sharpsat-td"
    aspmc_logger.setLevel("ERROR")

    # ------------------------------------------------------------------
    # Main query loop — one row per iaction in the test CSV
    # ------------------------------------------------------------------
    with open(input_csv, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        row_num = 0

        for row in reader:
            row_num += 1
            evidence      = {}
            interventions = {}
            queries       = []

            # --- Build evidence ---
            # Observed action
            if row['action'] not in actions:
                print(f"[Warning] Observed action '{row['action']}' not in actions list — skipping row {row_num}", flush=True)
                continue

            evidence[f"action({row['action']})"] = False  # True -> phase=False

            for col in ['curr_lane', 'free_E', 'free_NE', 'free_NW',
                        'free_SE', 'free_SW', 'free_W', 'latent_collision']:
                evidence[col] = (row[col] != "True")  # phase: False if value=="True"

            # --- Build intervention ---
            iaction = row['iaction']
            if iaction not in actions:
                print(f"[Warning] iaction '{iaction}' not in actions list — writing placeholder row", flush=True)
                with open(output_csv, 'a', newline='') as out:
                    csv.writer(out).writerow([
                        row['action'], row['curr_lane'], row['free_E'], row['free_NE'],
                        row['free_NW'], row['free_SE'], row['free_SW'], row['free_W'],
                        row['orig_label_lc'], row['latent_collision'],
                        row['iaction'], 1.0, 0.0, fold
                    ])
                continue

            action_idx = actions.index(iaction)
            iaction_str = ""
            for i, a in enumerate(actions):
                val   = iaction_truth[action_idx][i]
                phase = (val != "True")
                interventions[f"action({a})"] = phase
                iaction_str += f" -i action\\({a}\\),{val}"

            queries.append("latent_collision")

            # Debug call string (mirrors original WhatIf format)
            whatif_call = (
                f"WhatIf -q latent_collision"
                f" -e action\\({row['action']}\\),True"
                f" -e curr_lane,{row['curr_lane']}"
                f" -e free_E,{row['free_E']}"
                f" -e free_NE,{row['free_NE']}"
                f" -e free_NW,{row['free_NW']}"
                f" -e free_SE,{row['free_SE']}"
                f" -e free_SW,{row['free_SW']}"
                f" -e free_W,{row['free_W']}"
                f" -e latent_collision,{row['latent_collision']}"
                f"{iaction_str} {input_cbn}"
            )
            print(whatif_call, flush=True)

            # --- Execute query ---
            try:
                t0 = time.time()
                output_query = program.single_query(
                    interventions, evidence, queries,
                    strategy=config.config["knowledge_compiler"]
                )
                elapsed = time.time() - t0
                prob = float(output_query[0])

                print(f"Row {row_num} | P(latent_collision)={prob:.6f} | {elapsed:.2f}s", flush=True)

                with open(output_csv, 'a', newline='') as out:
                    csv.writer(out).writerow([
                        row['action'], row['curr_lane'], row['free_E'], row['free_NE'],
                        row['free_NW'], row['free_SE'], row['free_SW'], row['free_W'],
                        row['orig_label_lc'], row['latent_collision'],
                        iaction, prob, elapsed, fold
                    ])

                with open(elapsed_time_path, 'a') as f:
                    f.write(f"{elapsed}\n")

            except Exception as e:
                print(f"[Error] Query failed at row {row_num}: {type(e).__name__}: {e}", flush=True)
                print(f"  Evidence    : {evidence}",      flush=True)
                print(f"  Intervention: {interventions}", flush=True)
                print(f"  Query       : {queries}",       flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 run_WhatIf_V3.py <input.csv> <input.pl> <output.csv> "
              "<models_subdir> <output_actions_found> <fold_index>")
        sys.exit(1)

    main(
        input_csv            = sys.argv[1],
        input_cbn            = sys.argv[2],
        output_csv           = sys.argv[3],
        models_subdir        = sys.argv[4],
        output_actions_found = sys.argv[5],
        fold                 = sys.argv[6]
    )
