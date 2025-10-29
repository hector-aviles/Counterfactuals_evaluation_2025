#!/usr/bin/env python3
import csv
import time
from counterfactuals.counterfactualprogram import CounterfactualProgram
import aspmc.config as config
from aspmc.main import logger as aspmc_logger
import re
import os

def run_whatif(input_csv, input_cbn, models_subdir, output_actions_found, fold):
    """
    Runs all counterfactual queries in a single fold and returns a list of results.
    Each result is a tuple:
    (action, curr_lane, free_E, free_NE, free_NW, free_SE, free_SW, free_W,
     orig_label_lc, latent_collision, iaction, probability, elapsed_time, fold)
    """
    # Load actions
    actions_list = ["change_to_left", "change_to_right", "cruise", "keep", "swerve_left", "swerve_right"]
    actions = []

    try:
        with open(input_cbn, 'r') as file:
            content = file.read()
            actions_found = [a for a in actions_list if re.search(rf"{a}", content)]
            actions = list(set(actions_found))
            with open(output_actions_found, "a") as f:
                f.write(str(actions) + "\n")
    except FileNotFoundError:
        print(f"Warning: Could not read file {input_cbn}, using default actions only")

    # Generate truth tuples
    iaction_truth = []
    for i in range(len(actions)):
        truth_values = ['False'] * len(actions)
        truth_values[i] = 'True'
        iaction_truth.append(tuple(truth_values))

    # Load the ProbLog program
    program = CounterfactualProgram("", [input_cbn])
    config.config["knowledge_compiler"] = "sharpsat-td"
    aspmc_logger.setLevel("ERROR")

    results = []
    elapsed_times = []

    with open(input_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            # Evidence
            evidence = {}
            for key in ['action', 'curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W', 'latent_collision']:
                value = row[key]
                phase = False if value == "True" else True
                if key == 'action':
                    evidence[f'action({value})'] = phase
                else:
                    evidence[key] = phase

            # Interventions
            iaction = row['iaction']
            try:
                action_idx = actions.index(iaction)
            except ValueError:
                prob = 1.0
                elapsed_time = 0.0
                results.append(tuple([
                    row['action'], row['curr_lane'], row['free_E'], row['free_NE'], row['free_NW'],
                    row['free_SE'], row['free_SW'], row['free_W'], row['orig_label_lc'], row['latent_collision'],
                    iaction, prob, elapsed_time, fold
                ]))
                continue

            interventions = {f'action({a})': (False if v == "True" else True)
                             for a, v in zip(actions, iaction_truth[action_idx])}

            # Query
            queries = ["latent_collision"]
            start_time = time.time()
            output_query = program.single_query(interventions, evidence, queries, strategy=config.config["knowledge_compiler"])
            end_time = time.time()
            elapsed_time = end_time - start_time
            prob = float(output_query[0])

            elapsed_times.append(elapsed_time)

            results.append(tuple([
                row['action'], row['curr_lane'], row['free_E'], row['free_NE'], row['free_NW'],
                row['free_SE'], row['free_SW'], row['free_W'], row['orig_label_lc'], row['latent_collision'],
                iaction, prob, elapsed_time, fold
            ]))

    return results, elapsed_times

