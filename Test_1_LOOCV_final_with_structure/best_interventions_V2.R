#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import csv
from collections import Counter, defaultdict

# Path to no_crashes dataset used to decide potential_crash_before/after
NO_CRASHES_PATH = "./Shared_CSVs/no_crashes.csv"

# Tolerance for floating equality when detecting ties (equal probabilities)
EPS = 1e-12

def make_key_from_df(df, cols):
    """Return a Series of string keys by concatenating values in cols (all cast to str)."""
    return df[cols].astype(str).agg("_".join, axis=1)

def process_rep_perc(rep_num, perc, no_crashes_df):
    base_dir = os.getcwd()
    cbn_dir = os.path.join(base_dir, f"rep_{rep_num}", perc, "cBNs")
    input_csv = os.path.join(cbn_dir, "twin_networks_results.csv")
    bi_csv = os.path.join(cbn_dir, "best_interventions.csv")
    ds_csv = os.path.join(cbn_dir, "data_sorted.csv")
    ct_txt = os.path.join(cbn_dir, "contingency_table.txt")

    if not os.path.exists(input_csv):
        print(f"[Warning] Input file not found: {input_csv}. Skipping.", flush=True)
        return

    print(f"\nProcessing rep: {rep_num} percentage: {perc}")
    print("Loading:", input_csv)

    # Read results
    df = pd.read_csv(input_csv, dtype=str)  # read as strings first
    # convert numeric columns
    if 'probability' in df.columns:
        df['probability'] = pd.to_numeric(df['probability'], errors='coerce')
    else:
        raise RuntimeError("Input CSV must contain 'probability' column.")
    if 'elapsed_time' in df.columns:
        df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce')
    # Ensure group_id exists
    if 'group_id' not in df.columns:
        raise RuntimeError("Input CSV must contain 'group_id' column.")

    # --- Sort and compute ranking (dense rank by probability within group) ---
    df = df.sort_values(['group_id', 'probability'], ascending=[True, True]).reset_index(drop=True)
    # dense rank per group
    df['ranking'] = df.groupby('group_id')['probability'].rank(method='dense', ascending=True).astype(int)

    # Prepare ds_temp columns: same logic as your R code:
    # remove latent collision and iaction/probability/elapsed_time/group_id/ranking for existence checks.
    # Identify columns to exclude for state-key
    exclude_cols = set(['iaction', 'probability', 'elapsed_time', 'group_id', 'ranking', 'best_intervention'])
    # also exclude any label or latent collision as in your R code
    for to_ex in ['latent_collision', 'labeled_lc', 'orig_label_lc']:
        if to_ex in df.columns:
            exclude_cols.add(to_ex)

    state_cols = [c for c in df.columns if c not in exclude_cols]

    # Ensure 'action' is present in state_cols (we will overwrite it for after intervention)
    if 'action' not in state_cols:
        # if no 'action' column found, try 'iaction' presence etc.
        raise RuntimeError("'action' column not found in input CSV. Required for state key generation.")

    # Create keys for before intervention (action = observed action)
    before_key = make_key_from_df(df[state_cols], state_cols)

    # Create lookup set for no_crashes keys
    # Read no_crashes_df columns matching state_cols (if no_crashes doesn't have same set, try intersection)
    no_crashes_cols = [c for c in state_cols if c in no_crashes_df.columns]
    if len(no_crashes_cols) < len(state_cols):
        # If some state columns are missing in no_crashes, use intersection but warn
        missing = set(state_cols) - set(no_crashes_cols)
        if len(missing) > 0:
            print(f"[Warning] no_crashes.csv missing columns {missing}. Matching on intersection of columns.", flush=True)

    if len(no_crashes_cols) == 0:
        raise RuntimeError("no_crashes.csv has no columns in common with state columns; cannot compute potential_crash_*.")

    no_crashes_key = make_key_from_df(no_crashes_df[no_crashes_cols], no_crashes_cols)
    no_crashes_set = set(no_crashes_key.values)

    # Map before membership
    # To compare keys we must ensure same columns order/selection. Build comparable keys from df using no_crashes_cols.
    before_key_match = make_key_from_df(df[no_crashes_cols], no_crashes_cols)
    df['potential_crash_before_intervention'] = np.where(before_key_match.isin(no_crashes_set), 'False', 'True')

    # For after intervention, replace 'action' with 'iaction' and recompute keys
    df_after_keys = df[no_crashes_cols].copy()
    if 'action' in df_after_keys.columns and 'iaction' in df.columns:
        df_after_keys['action'] = df['iaction'].astype(str)
    after_key = make_key_from_df(df_after_keys, no_crashes_cols)
    df['potential_crash_after_intervention'] = np.where(after_key.isin(no_crashes_set), 'False', 'True')

    # Restore True for rows where action == iaction (original action)
    if 'action' in df.columns:
        df.loc[df['action'] == df['iaction'], 'potential_crash_after_intervention'] = 'True'

    # Initialize best_intervention column
    df['best_intervention'] = ''

    # --- Determine best candidates (ranking == 1) and apply random tie-breaking ---
    # Candidate rows are those with ranking == 1 (lowest probability)
    candidates = df[df['ranking'] == 1].copy()

    # Count how many candidate rows per group (ties)
    candidate_counts = candidates.groupby('group_id').size().rename('num_best').reset_index()

    # For reproducible tie-breaking, seed with a combination of rep and perc
    seed_int = hash((rep_num, perc)) & 0xffffffff
    rng = np.random.default_rng(seed_int)

    # For each group, pick one random candidate (if multiple)
    chosen_idx = []
    # We'll also build best_int dataframe like R's best_int (selected rows)
    best_int_rows = []
    for gid, group_df in candidates.groupby('group_id'):
        if len(group_df) == 1:
            chosen = group_df.index[0]
        else:
            chosen = rng.choice(group_df.index.values, size=1)[0]
        chosen_idx.append(chosen)
        best_int_rows.append(df.loc[chosen].copy())

    if len(best_int_rows) > 0:
        best_int = pd.DataFrame(best_int_rows).reset_index(drop=True)
    else:
        best_int = pd.DataFrame(columns=df.columns)

    # Mark best intervention in df
    # match by group_id and iaction
    # best_int contains one chosen row per group (selected best action)
    for _, row_sel in best_int.iterrows():
        mask = (df['group_id'] == row_sel['group_id']) & (df['iaction'] == row_sel['iaction'])
        df.loc[mask, 'best_intervention'] = '*'

    # Save best_interventions (selected ones) to CSV (drop helper columns if needed)
    # Match R's best_int selection: remove some columns as they did (-group_id, -ranking, -elapsed_time, -best_intervention)
    to_drop = [c for c in ['group_id', 'ranking', 'elapsed_time', 'best_intervention'] if c in best_int.columns]
    best_int_save = best_int.drop(columns=to_drop, errors='ignore')
    best_int_save.to_csv(bi_csv, index=False)
    print("Saved best interventions to", bi_csv)

    # Save full annotated sorted data (similar to data_sorted in R)
    # keep the same column order if possible
    df.to_csv(ds_csv, index=False)
    print("Saved sorted data to", ds_csv)

    # --------------------------
    # Build contingency report (text file) matching R format and labels
    # --------------------------
    with open(ct_txt, "w") as out:
        def wln(s=""):
            out.write(s + "\n")

        # Crash and no-crash partitions based on potential_crash_before_intervention
        crash_data = df[df['potential_crash_before_intervention'] == 'True']
        no_crash_data = df[df['potential_crash_before_intervention'] == 'False']

        # Header
        wln("=== CRASH SECTION ===")
        wln("(Data from rows where potential_crash_before_intervention is True)\n")

        # Counting results: number of groups with exactly i best interventions (i=1..6)
        # we need counts over groups of how many ranking==1 rows they had (ties)
        group_best_counts = (candidates
                             .groupby('group_id')
                             .size()
                             .rename('count_ranking1')
                             .reset_index())

        # For crash_data we only consider those groups that are in crash_data and ranking==1 rows in crash_data
        crash_ranking1 = (crash_data[crash_data['ranking'] == 1]
                          .groupby('group_id')
                          .size()
                          .rename('n')
                          .reset_index())

        # Count how many groups have exactly i best interventions in crash_data
        for i in range(1, 7):
            num = int((crash_ranking1['n'] == i).sum()) if not crash_ranking1.empty else 0
            wln(f"Number of groups with exactly {i} best interventions: {num}")

        # Action counts for crash section (only ranking == 1 rows)
        action_counts_crash = (crash_data[crash_data['ranking'] == 1]
                               .groupby('iaction', as_index=False)
                               .agg(total_selected=('iaction', 'size'),
                                    safe_count=('potential_crash_after_intervention', lambda x: (x == 'False').sum()),
                                    unsafe_count=('potential_crash_after_intervention', lambda x: (x == 'True').sum()))
                               ).sort_values('iaction')

        wln("\nNumber of actions selected for each intervention type:")
        if action_counts_crash.shape[0] == 0:
            wln("(none)\n")
        else:
            for _, r in action_counts_crash.iterrows():
                wln(f"{r['iaction']} : Selected {int(r['total_selected'])} times, Safe {int(r['safe_count'])} times, Unsafe {int(r['unsafe_count'])} times")

        # Totals
        total_safe_crash = int((crash_data[crash_data['ranking'] == 1]['potential_crash_after_intervention'] == 'False').sum())
        total_unsafe_crash = int((crash_data[crash_data['ranking'] == 1]['potential_crash_after_intervention'] == 'True').sum())

        wln("\nTotal safe/unsafe actions:")
        wln(f"Total safe actions: {total_safe_crash}")
        wln(f"Total unsafe actions: {total_unsafe_crash}")

        # Transition matrix for crash section (before -> after), using ranking==1 rows
        wln("\nTransition matrix (before -> after):")
        # build contingency using ranking==1 rows
        cr = crash_data[crash_data['ranking'] == 1]
        # Ensure keys exist
        before_levels = ['True', 'False']
        after_levels = ['True', 'False']
        trans = pd.crosstab(cr['potential_crash_before_intervention'], cr['potential_crash_after_intervention']).reindex(index=before_levels, columns=after_levels, fill_value=0)
        # Write two lines as R did (True->True, True->False)
        true_true = int(trans.loc['True','True']) if ('True' in trans.index and 'True' in trans.columns) else 0
        true_false = int(trans.loc['True','False']) if ('True' in trans.index and 'False' in trans.columns) else 0
        # number of groups contributing:
        n_tt_groups = int(cr[(cr['potential_crash_before_intervention']=='True') & (cr['potential_crash_after_intervention']=='True')]['group_id'].nunique())
        n_tf_groups = int(cr[(cr['potential_crash_before_intervention']=='True') & (cr['potential_crash_after_intervention']=='False')]['group_id'].nunique())
        wln(f"    True -> True: {true_true} (from {n_tt_groups} groups)")
        wln(f"    True -> False: {true_false} (from {n_tf_groups} groups)")

        # Detailed transition matrices by number of ties for CRASH SECTION
        wln("\nDetailed transition matrices by number of ties (CRASH SECTION):")
        # For i in 1..6
        # For groups with exactly i tied best interventions (i.e., groups with exactly i ranking==1 rows),
        # compute transition counts (before->after)
        if cr.empty:
            wln("(no ranking==1 rows in crash section)\n")
        else:
            # compute number of ranking==1 rows per group (in crash_data ranking==1 only)
            cr_group_counts = cr.groupby('group_id').size().rename('nrows').reset_index()
            for i in range(1,7):
                groups_with_i = cr_group_counts[cr_group_counts['nrows'] == i]['group_id'].tolist()
                if len(groups_with_i) == 0:
                    continue
                tied_data = cr[cr['group_id'].isin(groups_with_i)]
                wln(f"\nFor groups with {i} tied best interventions:")
                wln("  - Transition matrix (before -> after):")
                tr = pd.crosstab(tied_data['potential_crash_before_intervention'], tied_data['potential_crash_after_intervention']).reindex(index=before_levels, columns=after_levels, fill_value=0)
                t_tt = int(tr.loc['True','True']) if ('True' in tr.index and 'True' in tr.columns) else 0
                t_tf = int(tr.loc['True','False']) if ('True' in tr.index and 'False' in tr.columns) else 0
                # group counts
                group_counts = tied_data.groupby(['potential_crash_before_intervention','potential_crash_after_intervention'])['group_id'].nunique().reset_index(name='n_groups')
                def get_group_count(b,a):
                    found = group_counts[(group_counts['potential_crash_before_intervention']==b) & (group_counts['potential_crash_after_intervention']==a)]
                    return int(found['n_groups'].iloc[0]) if not found.empty else 0
                g_tt = get_group_count('True','True')
                g_tf = get_group_count('True','False')
                wln(f"    True -> True: {t_tt} (from {g_tt} groups)")
                wln(f"    True -> False: {t_tf} (from {g_tf} groups)")

        # NO CRASH SECTION
        wln("\n\n=== NO CRASH SECTION ===")
        wln("(Data from rows where potential_crash_before_intervention is False)\n")

        # Counting results for no_crash section (ranking==1)
        nc = no_crash_data[no_crash_data['ranking'] == 1]
        nc_group_counts = nc.groupby('group_id').size().rename('n').reset_index()
        for i in range(1,7):
            num = int((nc_group_counts['n'] == i).sum()) if not nc_group_counts.empty else 0
            wln(f"Number of groups with exactly {i} best interventions: {num}")

        # Action counts for no_crash section
        action_counts_no_crash = (nc
                                  .groupby('iaction', as_index=False)
                                  .agg(total_selected=('iaction','size'),
                                       safe_count=('potential_crash_after_intervention', lambda x: (x == 'False').sum()),
                                       unsafe_count=('potential_crash_after_intervention', lambda x: (x == 'True').sum()))
                                  ).sort_values('iaction')

        wln("\nNumber of actions selected for each intervention type:")
        if action_counts_no_crash.shape[0] == 0:
            wln("(none)\n")
        else:
            for _, r in action_counts_no_crash.iterrows():
                wln(f"{r['iaction']} : Selected {int(r['total_selected'])} times, Safe {int(r['safe_count'])} times, Unsafe {int(r['unsafe_count'])} times")

        total_safe_no_crash = int((nc['potential_crash_after_intervention'] == 'False').sum())
        total_unsafe_no_crash = int((nc['potential_crash_after_intervention'] == 'True').sum())

        wln("\nTotal safe/unsafe actions:")
        wln(f"Total safe actions: {total_safe_no_crash}")
        wln(f"Total unsafe actions: {total_unsafe_no_crash}")

        # Transition matrix for no_crash section (ranking==1 rows)
        wln("\nTransition matrix (before -> after):")
        tr_nc = pd.crosstab(nc['potential_crash_before_intervention'], nc['potential_crash_after_intervention']).reindex(index=before_levels, columns=after_levels, fill_value=0)
        f_t = int(tr_nc.loc['False','True']) if ('False' in tr_nc.index and 'True' in tr_nc.columns) else 0
        f_f = int(tr_nc.loc['False','False']) if ('False' in tr_nc.index and 'False' in tr_nc.columns) else 0
        g_ft = int(nc[(nc['potential_crash_before_intervention']=='False') & (nc['potential_crash_after_intervention']=='True')]['group_id'].nunique())
        g_ff = int(nc[(nc['potential_crash_before_intervention']=='False') & (nc['potential_crash_after_intervention']=='False')]['group_id'].nunique())
        wln(f"    False -> True: {f_t} (from {g_ft} groups)")
        wln(f"    False -> False: {f_f} (from {g_ff} groups)")

        # Detailed transition matrices by number of ties (NO CRASH SECTION)
        wln("\n\nDetailed transition matrices by number of ties (NO CRASH SECTION):")
        if nc.empty:
            wln("(no ranking==1 rows in no-crash section)\n")
        else:
            nc_group_counts = nc.groupby('group_id').size().rename('nrows').reset_index()
            for i in range(1,7):
                groups_with_i = nc_group_counts[nc_group_counts['nrows']==i]['group_id'].tolist()
                if len(groups_with_i) == 0:
                    continue
                tied_data = nc[nc['group_id'].isin(groups_with_i)]
                wln(f"\nFor groups with {i} tied best interventions:")
                wln("  - Transition matrix (before -> after):")
                tr = pd.crosstab(tied_data['potential_crash_before_intervention'], tied_data['potential_crash_after_intervention']).reindex(index=before_levels, columns=after_levels, fill_value=0)
                aa = int(tr.loc['False','True']) if ('False' in tr.index and 'True' in tr.columns) else 0
                bb = int(tr.loc['False','False']) if ('False' in tr.index and 'False' in tr.columns) else 0
                group_counts = tied_data.groupby(['potential_crash_before_intervention','potential_crash_after_intervention'])['group_id'].nunique().reset_index(name='n_groups')
                def get_group_count2(b,a):
                    found = group_counts[(group_counts['potential_crash_before_intervention']==b) & (group_counts['potential_crash_after_intervention']==a)]
                    return int(found['n_groups'].iloc[0]) if not found.empty else 0
                g_at = get_group_count2('False','True')
                g_af = get_group_count2('False','False')
                wln(f"    False -> True: {aa} (from {g_at} groups)")
                wln(f"    False -> False: {bb} (from {g_af} groups)")

        # Final summary section (random selection outcomes from best_int)
        wln("\n\n=== FINAL SUMMARY ===")
        # best_int contains chosen best rows per group
        bt = best_int.copy()
        if bt.empty:
            wln("Random selection - no best_int rows found.")
        else:
            random_true_true = int(((bt['potential_crash_before_intervention']=='True') & (bt['potential_crash_after_intervention']=='True')).sum())
            random_false_false = int(((bt['potential_crash_before_intervention']=='False') & (bt['potential_crash_after_intervention']=='False')).sum())
            random_true_false = int(((bt['potential_crash_before_intervention']=='True') & (bt['potential_crash_after_intervention']=='False')).sum())
            random_false_true = int(((bt['potential_crash_before_intervention']=='False') & (bt['potential_crash_after_intervention']=='True')).sum())
            wln(f"Random selection - Crash before (True) and after intervention (True): {random_true_true}")
            wln(f"Random selection - Crash before (False) and after intervention (False): {random_false_false}")
            wln(f"Random selection - Crash before (True) and after intervention (False): {random_true_false}")
            wln(f"Random selection - Crash before (False) and after intervention (True): {random_false_true}")

    print("Saved contingency table to", ct_txt)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 summarize_twin_networks.py <num_reps> <percentages_comma_separated>")
        print("Example: python3 summarize_twin_networks.py 5 01,25,50,75,90")
        sys.exit(1)

    num_reps = int(sys.argv[1])
    percentages = sys.argv[2].split(",")

    # Load no_crashes
    if not os.path.exists(NO_CRASHES_PATH):
        print(f"[Error] no_crashes file not found at {NO_CRASHES_PATH}. Please update NO_CRASHES_PATH in the script or place the file there.")
        sys.exit(1)
    no_crashes_df = pd.read_csv(NO_CRASHES_PATH, dtype=str)

    for rep in range(1, num_reps + 1):
        for perc in percentages:
            try:
                process_rep_perc(rep, perc, no_crashes_df)
            except Exception as e:
                print(f"[Error] processing rep {rep} perc {perc}: {e}", flush=True)

    print("\nAll processing done.")

if __name__ == "__main__":
    main()

