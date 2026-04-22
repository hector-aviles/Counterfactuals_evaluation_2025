#!/usr/bin/env python3
# =============================================================================
# best_interventions_hc_sensitivity.py
#
# Step 3: safety labeling and contingency analysis for HC+BIC sensitivity
# combinations. Adapted from best_interventions_with_frequency.py.
#
# Input per combination:
#   <combo_dir>/twin_networks_results_all.csv
#
# Output per combination:
#   <combo_dir>/best_interventions.csv
#   <combo_dir>/data_sorted.csv
#   <combo_dir>/contingency_table.txt
#   <combo_dir>/contingency_table_atomized.txt
#
# Usage:
#   python3 best_interventions_hc_sensitivity.py [--combos dir1,dir2,...] [--all]
#
#   --all     : process all hc_* directories found in cwd (default)
#   --combos  : comma-separated list of specific combo dirs to process
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
import csv
from collections import defaultdict

NO_CRASHES_PATH  = "./Shared_CSVs/no_crashes.csv"
COMPLETE_DB_PATH = "./Shared_CSVs/complete_DB_discrete.csv"
EPS              = 1e-12


# =============================================================================
# Shared utilities (unchanged from original)
# =============================================================================
def make_key_from_df(df, cols):
    return df[cols].astype(str).agg("_".join, axis=1)


def load_frequency_data():
    print("Loading frequency data from:", COMPLETE_DB_PATH)
    if not os.path.exists(COMPLETE_DB_PATH):
        print("[Warning] Complete DB not found. Frequency column will be NA.")
        return None

    freq_df = pd.read_csv(COMPLETE_DB_PATH, dtype=str)
    state_cols = ['curr_lane', 'free_E', 'free_NE', 'free_NW',
                  'free_SE', 'free_SW', 'free_W']
    available = [c for c in state_cols if c in freq_df.columns]

    if 'action' not in freq_df.columns or not available:
        print("[Warning] Required columns missing in frequency data.")
        return None

    print(f"State columns for frequency: {available}")
    freq_df['freq_key'] = make_key_from_df(freq_df[['action'] + available],
                                            ['action'] + available)
    counts = freq_df.groupby('freq_key').size().reset_index(name='frequency')
    return counts.set_index('freq_key')['frequency'].to_dict()


def add_frequency_column(df, frequency_dict):
    if frequency_dict is None:
        df['frequency'] = 'NA'
        return df

    state_cols = ['curr_lane', 'free_E', 'free_NE', 'free_NW',
                  'free_SE', 'free_SW', 'free_W']
    available = [c for c in state_cols if c in df.columns]

    if 'action' not in df.columns or not available:
        df['frequency'] = 'NA'
        return df

    df['freq_key'] = make_key_from_df(df[['action'] + available],
                                       ['action'] + available)
    df['frequency'] = 'NA'
    mask = df['ranking'] == 1
    df.loc[mask, 'frequency'] = df.loc[mask, 'freq_key'].map(
        lambda x: str(frequency_dict.get(x, 0)) if pd.notna(x) else '0'
    )
    df.drop('freq_key', axis=1, inplace=True, errors='ignore')
    return df


# =============================================================================
# Contingency table writers (unchanged logic, parameterised on output handle)
# =============================================================================
def write_contingency(df, best_int, out):
    def wln(s=""): out.write(s + "\n")

    before_levels = ['True', 'False']
    after_levels  = ['True', 'False']

    crash_data    = df[df['potential_crash_before_intervention'] == 'True']
    no_crash_data = df[df['potential_crash_before_intervention'] == 'False']

    # ── CRASH SECTION ────────────────────────────────────────────────────────
    wln("=== CRASH SECTION ===")
    wln("(Data from rows where potential_crash_before_intervention is True)\n")

    cr = crash_data[crash_data['ranking'] == 1]
    cr_group_counts = cr.groupby('group_id').size().rename('n').reset_index()
    for i in range(1, 7):
        num = int((cr_group_counts['n'] == i).sum()) if not cr_group_counts.empty else 0
        wln(f"Number of groups with exactly {i} best interventions: {num}")

    ac = (cr.groupby('iaction', as_index=False)
            .agg(total_selected=('iaction', 'size'),
                 safe_count=('potential_crash_after_intervention', lambda x: (x=='False').sum()),
                 unsafe_count=('potential_crash_after_intervention', lambda x: (x=='True').sum()))
            .sort_values('iaction'))
    wln("\nNumber of actions selected for each intervention type:")
    if ac.empty:
        wln("(none)\n")
    else:
        for _, r in ac.iterrows():
            wln(f"{r['iaction']} : Selected {int(r['total_selected'])} times, "
                f"Safe {int(r['safe_count'])} times, Unsafe {int(r['unsafe_count'])} times")

    wln(f"\nTotal safe/unsafe actions:")
    wln(f"Total safe actions: {int((cr['potential_crash_after_intervention']=='False').sum())}")
    wln(f"Total unsafe actions: {int((cr['potential_crash_after_intervention']=='True').sum())}")

    wln("\nTransition matrix (before -> after):")
    if not cr.empty:
        tr = pd.crosstab(cr['potential_crash_before_intervention'],
                         cr['potential_crash_after_intervention']
                         ).reindex(index=before_levels, columns=after_levels, fill_value=0)
        tt = int(tr.loc['True','True'])   if ('True'  in tr.index and 'True'  in tr.columns) else 0
        tf = int(tr.loc['True','False'])  if ('True'  in tr.index and 'False' in tr.columns) else 0
        g_tt = int(cr[(cr['potential_crash_before_intervention']=='True') &
                       (cr['potential_crash_after_intervention']=='True')]['group_id'].nunique())
        g_tf = int(cr[(cr['potential_crash_before_intervention']=='True') &
                       (cr['potential_crash_after_intervention']=='False')]['group_id'].nunique())
        wln(f"    True -> True: {tt} (from {g_tt} groups)")
        wln(f"    True -> False: {tf} (from {g_tf} groups)")

    wln("\nDetailed transition matrices by number of ties (CRASH SECTION):")
    if cr.empty:
        wln("(no ranking==1 rows in crash section)\n")
    else:
        gc = cr.groupby('group_id').size().rename('nrows').reset_index()
        for i in range(1, 7):
            groups_i = gc[gc['nrows'] == i]['group_id'].tolist()
            if not groups_i: continue
            td = cr[cr['group_id'].isin(groups_i)]
            wln(f"\nFor groups with {i} tied best interventions:")
            wln("  - Transition matrix (before -> after):")
            tr = pd.crosstab(td['potential_crash_before_intervention'],
                             td['potential_crash_after_intervention']
                             ).reindex(index=before_levels, columns=after_levels, fill_value=0)
            t_tt = int(tr.loc['True','True'])  if ('True' in tr.index and 'True'  in tr.columns) else 0
            t_tf = int(tr.loc['True','False']) if ('True' in tr.index and 'False' in tr.columns) else 0
            gc2 = td.groupby(['potential_crash_before_intervention',
                              'potential_crash_after_intervention'])['group_id'].nunique().reset_index(name='n')
            def _g(b, a):
                f = gc2[(gc2['potential_crash_before_intervention']==b) &
                        (gc2['potential_crash_after_intervention']==a)]
                return int(f['n'].iloc[0]) if not f.empty else 0
            wln(f"    True -> True: {t_tt} (from {_g('True','True')} groups)")
            wln(f"    True -> False: {t_tf} (from {_g('True','False')} groups)")

    # ── NO CRASH SECTION ─────────────────────────────────────────────────────
    wln("\n\n=== NO CRASH SECTION ===")
    wln("(Data from rows where potential_crash_before_intervention is False)\n")

    nc = no_crash_data[no_crash_data['ranking'] == 1]
    nc_gc = nc.groupby('group_id').size().rename('n').reset_index()
    for i in range(1, 7):
        num = int((nc_gc['n'] == i).sum()) if not nc_gc.empty else 0
        wln(f"Number of groups with exactly {i} best interventions: {num}")

    ac_nc = (nc.groupby('iaction', as_index=False)
               .agg(total_selected=('iaction', 'size'),
                    safe_count=('potential_crash_after_intervention', lambda x: (x=='False').sum()),
                    unsafe_count=('potential_crash_after_intervention', lambda x: (x=='True').sum()))
               .sort_values('iaction'))
    wln("\nNumber of actions selected for each intervention type:")
    if ac_nc.empty:
        wln("(none)\n")
    else:
        for _, r in ac_nc.iterrows():
            wln(f"{r['iaction']} : Selected {int(r['total_selected'])} times, "
                f"Safe {int(r['safe_count'])} times, Unsafe {int(r['unsafe_count'])} times")

    wln(f"\nTotal safe/unsafe actions:")
    wln(f"Total safe actions: {int((nc['potential_crash_after_intervention']=='False').sum())}")
    wln(f"Total unsafe actions: {int((nc['potential_crash_after_intervention']=='True').sum())}")

    wln("\nTransition matrix (before -> after):")
    if not nc.empty:
        tr_nc = pd.crosstab(nc['potential_crash_before_intervention'],
                            nc['potential_crash_after_intervention']
                            ).reindex(index=before_levels, columns=after_levels, fill_value=0)
        ft = int(tr_nc.loc['False','True'])  if ('False' in tr_nc.index and 'True'  in tr_nc.columns) else 0
        ff = int(tr_nc.loc['False','False']) if ('False' in tr_nc.index and 'False' in tr_nc.columns) else 0
        g_ft = int(nc[(nc['potential_crash_before_intervention']=='False') &
                       (nc['potential_crash_after_intervention']=='True')]['group_id'].nunique())
        g_ff = int(nc[(nc['potential_crash_before_intervention']=='False') &
                       (nc['potential_crash_after_intervention']=='False')]['group_id'].nunique())
        wln(f"    False -> True: {ft} (from {g_ft} groups)")
        wln(f"    False -> False: {ff} (from {g_ff} groups)")

    wln("\n\nDetailed transition matrices by number of ties (NO CRASH SECTION):")
    if nc.empty:
        wln("(no ranking==1 rows in no-crash section)\n")
    else:
        gc_nc = nc.groupby('group_id').size().rename('nrows').reset_index()
        for i in range(1, 7):
            groups_i = gc_nc[gc_nc['nrows'] == i]['group_id'].tolist()
            if not groups_i: continue
            td = nc[nc['group_id'].isin(groups_i)]
            wln(f"\nFor groups with {i} tied best interventions:")
            wln("  - Transition matrix (before -> after):")
            tr = pd.crosstab(td['potential_crash_before_intervention'],
                             td['potential_crash_after_intervention']
                             ).reindex(index=before_levels, columns=after_levels, fill_value=0)
            aa = int(tr.loc['False','True'])  if ('False' in tr.index and 'True'  in tr.columns) else 0
            bb = int(tr.loc['False','False']) if ('False' in tr.index and 'False' in tr.columns) else 0
            gc3 = td.groupby(['potential_crash_before_intervention',
                              'potential_crash_after_intervention'])['group_id'].nunique().reset_index(name='n')
            def _g2(b, a):
                f = gc3[(gc3['potential_crash_before_intervention']==b) &
                        (gc3['potential_crash_after_intervention']==a)]
                return int(f['n'].iloc[0]) if not f.empty else 0
            wln(f"    False -> True: {aa} (from {_g2('False','True')} groups)")
            wln(f"    False -> False: {bb} (from {_g2('False','False')} groups)")

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    wln("\n\n=== FINAL SUMMARY ===")
    bt = best_int.copy()
    if bt.empty:
        wln("Random selection - no best_int rows found.")
    else:
        wln(f"Random selection - Crash before (True) and after intervention (True): "
            f"{int(((bt['potential_crash_before_intervention']=='True') & (bt['potential_crash_after_intervention']=='True')).sum())}")
        wln(f"Random selection - Crash before (False) and after intervention (False): "
            f"{int(((bt['potential_crash_before_intervention']=='False') & (bt['potential_crash_after_intervention']=='False')).sum())}")
        wln(f"Random selection - Crash before (True) and after intervention (False): "
            f"{int(((bt['potential_crash_before_intervention']=='True') & (bt['potential_crash_after_intervention']=='False')).sum())}")
        wln(f"Random selection - Crash before (False) and after intervention (True): "
            f"{int(((bt['potential_crash_before_intervention']=='False') & (bt['potential_crash_after_intervention']=='True')).sum())}")


def write_contingency_atomized(df, out):
    def wln(s=""): out.write(s + "\n")

    before_levels = ['True', 'False']
    after_levels  = ['True', 'False']

    def safe_freq(x):
        try:
            return int(x) if x != 'NA' else -1
        except (ValueError, TypeError):
            return -1

    crash_data    = df[df['potential_crash_before_intervention'] == 'True']
    no_crash_data = df[df['potential_crash_before_intervention'] == 'False']

    crash_zero    = crash_data[crash_data['frequency'].apply(safe_freq) == 0]
    crash_nonzero = crash_data[crash_data['frequency'].apply(safe_freq) >  0]
    nc_zero       = no_crash_data[no_crash_data['frequency'].apply(safe_freq) == 0]
    nc_nonzero    = no_crash_data[no_crash_data['frequency'].apply(safe_freq) >  0]

    def section(label, description, subset):
        wln(f"=== {label} ===")
        wln(f"({description})\n")
        r1 = subset[subset['ranking'] == 1]
        gc = r1.groupby('group_id').size().rename('n').reset_index()
        for i in range(1, 7):
            num = int((gc['n'] == i).sum()) if not gc.empty else 0
            wln(f"Number of groups with exactly {i} best interventions: {num}")

        ac = (r1.groupby('iaction', as_index=False)
                .agg(total_selected=('iaction', 'size'),
                     safe_count=('potential_crash_after_intervention', lambda x: (x=='False').sum()),
                     unsafe_count=('potential_crash_after_intervention', lambda x: (x=='True').sum()))
                .sort_values('iaction'))
        wln("\nNumber of actions selected for each intervention type:")
        if ac.empty:
            wln("(none)\n")
        else:
            for _, r in ac.iterrows():
                wln(f"{r['iaction']} : Selected {int(r['total_selected'])} times, "
                    f"Safe {int(r['safe_count'])} times, Unsafe {int(r['unsafe_count'])} times")

        wln(f"\nTotal safe/unsafe actions:")
        wln(f"Total safe actions: {int((r1['potential_crash_after_intervention']=='False').sum())}")
        wln(f"Total unsafe actions: {int((r1['potential_crash_after_intervention']=='True').sum())}")

        wln("\nTransition matrix (before -> after):")
        if not r1.empty:
            b_col = 'potential_crash_before_intervention'
            a_col = 'potential_crash_after_intervention'
            tr = pd.crosstab(r1[b_col], r1[a_col]).reindex(
                index=before_levels, columns=after_levels, fill_value=0)
            for bv in before_levels:
                for av in after_levels:
                    v = int(tr.loc[bv, av]) if (bv in tr.index and av in tr.columns) else 0
                    g = int(r1[(r1[b_col]==bv) & (r1[a_col]==av)]['group_id'].nunique())
                    wln(f"    {bv} -> {av}: {v} (from {g} groups)")
        else:
            wln("    (no data)")
        wln()

    section("CRASH SECTION - ZERO OCCURRENCES",
            "potential_crash_before_intervention is True and frequency = 0",
            crash_zero)
    section("CRASH SECTION - NON-ZERO OCCURRENCES (>=1)",
            "potential_crash_before_intervention is True and frequency >= 1",
            crash_nonzero)
    section("NO CRASH SECTION - ZERO OCCURRENCES",
            "potential_crash_before_intervention is False and frequency = 0",
            nc_zero)
    section("NO CRASH SECTION - NON-ZERO OCCURRENCES (>=1)",
            "potential_crash_before_intervention is False and frequency >= 1",
            nc_nonzero)


# =============================================================================
# Per-combination processor
# =============================================================================
def process_combo(combo_dir, no_crashes_df, frequency_dict):
    combo_name   = os.path.basename(os.path.abspath(combo_dir))
    input_csv    = os.path.join(combo_dir, "twin_networks_results_all.csv")
    bi_csv       = os.path.join(combo_dir, "best_interventions.csv")
    ds_csv       = os.path.join(combo_dir, "data_sorted.csv")
    ct_txt       = os.path.join(combo_dir, "contingency_table.txt")
    ct_atom_txt  = os.path.join(combo_dir, "contingency_table_atomized.txt")

    if not os.path.exists(input_csv):
        print(f"[Warning] Input not found: {input_csv} — skipping {combo_name}", flush=True)
        return

    print(f"\n{'='*60}")
    print(f"  Processing: {combo_name}")
    print(f"  Input: {input_csv}")
    print(f"{'='*60}")

    # Load results
    df = pd.read_csv(input_csv, dtype=str)
    if 'probability' not in df.columns:
        raise RuntimeError(f"'probability' column missing in {input_csv}")
    if 'group_id' not in df.columns:
        raise RuntimeError(f"'group_id' column missing in {input_csv}")

    df['probability']  = pd.to_numeric(df['probability'],  errors='coerce')
    df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce')
    df['group_id']     = pd.to_numeric(df['group_id'],     errors='coerce').astype(int)

    # Normalize crash columns to strings — pd.read_csv may load them as bool
    # if a previously saved data_sorted.csv is passed as input. All downstream
    # comparisons use string 'True'/'False'.
    for col in ['potential_crash_before_intervention',
                'potential_crash_after_intervention']:
        if col in df.columns:
            df[col] = df[col].map({True: 'True', False: 'False',
                                   'True': 'True', 'False': 'False',
                                   'true': 'True', 'false': 'False'})

    # --- Fold coverage diagnostics ---
    all_fold_ids   = sorted(df['group_id'].unique())
    n_folds_found  = len(all_fold_ids)
    rows_per_fold  = df.groupby('group_id').size()
    incomplete     = rows_per_fold[rows_per_fold < 6].index.tolist()

    # Always check against expected range 1..max_fold_id
    expected_folds = set(range(1, all_fold_ids[-1] + 1)) if all_fold_ids else set()
    missing_folds  = sorted(expected_folds - set(all_fold_ids))

    print(f"  Total rows in CSV        : {len(df)}")
    print(f"  Unique folds (group_ids) : {n_folds_found}")
    print(f"  Fold ID range            : {all_fold_ids[0]} – {all_fold_ids[-1]}")
    print(f"  Expected folds (1–{all_fold_ids[-1]})  : {len(expected_folds)}")
    if missing_folds:
        print(f"  [Warning] Missing fold IDs: {missing_folds}")
        print(f"  Rerun with: python3 run_hc_testing.py <combo_dir> --folds {missing_folds[0]}-{missing_folds[-1]}")
    else:
        print(f"  All fold IDs 1–{all_fold_ids[-1]} present — no gaps.")
    if incomplete:
        print(f"  [Warning] Folds with fewer than 6 rows (incomplete queries):")
        for fid in incomplete:
            print(f"    fold {fid}: {rows_per_fold[fid]} rows")
    else:
        print(f"  All present folds have exactly 6 rows.")

    # Rank by probability within group (ascending: lowest prob = rank 1)
    df = df.sort_values(['group_id', 'probability'], ascending=[True, True]).reset_index(drop=True)
    df['ranking'] = df.groupby('group_id')['probability'].rank(
        method='dense', ascending=True).astype(int)

    # Frequency column
    df = add_frequency_column(df, frequency_dict)

    # State columns for crash membership lookup
    state_cols = ['curr_lane', 'free_E', 'free_NE', 'free_NW',
                  'free_SE', 'free_SW', 'free_W']
    exclude_cols = {'iaction', 'probability', 'elapsed_time', 'group_id',
                    'ranking', 'best_intervention', 'frequency',
                    'latent_collision', 'labeled_lc', 'orig_label_lc'}
    used_state_cols = [c for c in df.columns if c not in exclude_cols]

    if 'action' not in used_state_cols:
        raise RuntimeError("'action' column required but not found.")

    # Build no_crashes lookup
    nc_cols = [c for c in used_state_cols if c in no_crashes_df.columns]
    if not nc_cols:
        raise RuntimeError("no_crashes.csv has no columns in common with state cols.")

    nc_key = set(make_key_from_df(no_crashes_df[nc_cols], nc_cols).values)

    before_key = make_key_from_df(df[nc_cols], nc_cols)
    df['potential_crash_before_intervention'] = np.where(
        before_key.isin(nc_key), 'False', 'True')

    df_after = df[nc_cols].copy()
    if 'action' in df_after.columns and 'iaction' in df.columns:
        df_after['action'] = df['iaction'].astype(str)
    after_key = make_key_from_df(df_after, nc_cols)
    df['potential_crash_after_intervention'] = np.where(
        after_key.isin(nc_key), 'False', 'True')

    # Restore True for self-intervention (action == iaction)
    if 'action' in df.columns:
        df.loc[df['action'] == df['iaction'], 'potential_crash_after_intervention'] = 'True'

    df['best_intervention'] = ''

    # Tie-breaking: pick one random best per group
    candidates = df[df['ranking'] == 1].copy()
    rng        = np.random.default_rng(hash(combo_name) & 0xffffffff)
    best_rows  = []
    for gid, grp in candidates.groupby('group_id'):
        idx = rng.choice(grp.index.values) if len(grp) > 1 else grp.index[0]
        best_rows.append(df.loc[idx].copy())
        df.loc[(df['group_id'] == gid) & (df['iaction'] == df.loc[idx, 'iaction']),
               'best_intervention'] = '*'

    best_int = pd.DataFrame(best_rows).reset_index(drop=True) if best_rows else pd.DataFrame(columns=df.columns)

    df_sorted = df.sort_values(['group_id', 'ranking']).reset_index(drop=True)

    # Save outputs
    drop_cols = [c for c in ['group_id', 'ranking', 'elapsed_time', 'best_intervention']
                 if c in best_int.columns]
    best_int.drop(columns=drop_cols, errors='ignore').to_csv(bi_csv, index=False)
    print(f"  Saved best interventions : {bi_csv}")

    df_sorted.to_csv(ds_csv, index=False)
    print(f"  Saved sorted data        : {ds_csv}")

    with open(ct_txt, 'w') as f:
        write_contingency(df_sorted, best_int, f)
    print(f"  Saved contingency table  : {ct_txt}")

    with open(ct_atom_txt, 'w') as f:
        write_contingency_atomized(df_sorted, f)
    print(f"  Saved atomized table     : {ct_atom_txt}")

    # --- Final group coverage report ---
    n_crash    = df_sorted[df_sorted['potential_crash_before_intervention'] == 'True' ]['group_id'].nunique()
    n_nocrash  = df_sorted[df_sorted['potential_crash_before_intervention'] == 'False']['group_id'].nunique()
    n_total    = df_sorted['group_id'].nunique()
    print(f"  Groups in crash section  : {n_crash}")
    print(f"  Groups in no-crash section: {n_nocrash}")
    print(f"  Total groups accounted   : {n_total}  (crash + no-crash = {n_crash + n_nocrash})")
    if n_crash + n_nocrash != n_total:
        overlap = df_sorted.groupby('group_id')['potential_crash_before_intervention'].nunique()
        mixed   = overlap[overlap > 1].index.tolist()
        print(f"  [Warning] {len(mixed)} group(s) appear in BOTH sections (mixed before label): {mixed}")
    if n_total < n_folds_found:
        print(f"  [Warning] {n_folds_found - n_total} fold(s) present in CSV but absent from contingency "
              f"(all rows filtered out during crash/no-crash assignment?)")


# =============================================================================
# Main
# =============================================================================
def main():
    # Parse arguments
    combo_dirs = []
    process_all = True
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--combos' and i + 1 < len(sys.argv):
            combo_dirs   = sys.argv[i+1].split(',')
            process_all  = False
            i += 2
        elif sys.argv[i] == '--all':
            process_all = True
            i += 1
        else:
            i += 1

    if process_all:
        combo_dirs = sorted([
            d for d in os.listdir('.')
            if os.path.isdir(d) and d.startswith('mmpc_')
            and os.path.exists(os.path.join(d, 'global_summary.csv'))
        ])
        if not combo_dirs:
            print("[Error] No mmpc_* directories with global_summary.csv found in current directory.")
            sys.exit(1)
        print(f"Found {len(combo_dirs)} combination directories: {combo_dirs}")

    # Load shared data once
    if not os.path.exists(NO_CRASHES_PATH):
        print(f"[Error] no_crashes.csv not found at {NO_CRASHES_PATH}")
        sys.exit(1)
    no_crashes_df  = pd.read_csv(NO_CRASHES_PATH, dtype=str)
    frequency_dict = load_frequency_data()

    # Process each combination
    for combo_dir in combo_dirs:
        try:
            process_combo(combo_dir, no_crashes_df, frequency_dict)
        except Exception as e:
            print(f"[Error] {combo_dir}: {e}", flush=True)

    print("\nAll combinations processed.")


if __name__ == "__main__":
    main()
