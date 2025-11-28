#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import csv
import pickle
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder

# Path to no_crashes dataset used to decide potential_crash_before/after
NO_CRASHES_PATH = "./Shared_CSVs/no_crashes.csv"
# Path to complete database for frequency lookup
COMPLETE_DB_PATH = "./Shared_CSVs/complete_DB_discrete.csv"

# Tolerance for floating equality when detecting ties (equal probabilities)
EPS = 1e-12

def load_nb_model(rep_num, percentage, fold_num):
    """Load Naive Bayes model from pickle file"""
    model_path = f"./rep_{rep_num}/{percentage}/NB/NB_fold_{fold_num}.pkl"
    
    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"[Error] Loading model {model_path}: {e}")
        return None

def compute_probability_for_iaction(model_data, test_data):
    """Compute P(iaction | state_vars) for each row using the given iaction from test data"""
    if model_data is None:
        return None
    
    model = model_data['model']
    encoder = model_data['encoder']
    
    # Prepare test features - use the same features as during training
    # latent_collision, free_E, free_NE, free_NW, free_SE, free_SW, free_W
    feature_columns = ["curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W", "latent_collision"]
    available_features = [col for col in feature_columns if col in test_data.columns]
    
    if not available_features:
        print("[Warning] No required features found in test data")
        return None
    
    X_test = test_data[available_features].copy()
    
    # Convert boolean-like strings to numerical values (same as training)
    for col in available_features:
        if X_test[col].dtype == 'object':
            X_test[col] = X_test[col].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
            try:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            except:
                pass
    
    # Fill NaN values
    X_test = X_test.fillna(0)
    
    # Compute probabilities for all classes
    try:
        all_probabilities = model.predict_proba(X_test)
        
        # For each row, get the probability of the specific iaction
        probabilities = []
        for i, row in test_data.iterrows():
            iaction = row['iaction']
            # Find the index of this iaction in the encoder classes
            try:
                action_index = np.where(encoder.classes_ == iaction)[0][0]
                prob = all_probabilities[i, action_index]
                probabilities.append(prob)
            except IndexError:
                # If iaction is not in the trained classes, use a small probability
                print(f"Warning: iaction '{iaction}' not found in trained classes. Using probability 0.0")
                probabilities.append(0.0)
        
        return {
            'probabilities': np.array(probabilities),
            'class_labels': encoder.classes_
        }
    except Exception as e:
        print(f"[Error] Computing probabilities: {e}")
        return None

def process_nb_fold(rep_num, percentage, fold_num, no_crashes_df, frequency_dict):
    """Process a single NB fold and generate results"""
    print(f"Processing NB rep {rep_num}, percentage {percentage}, fold {fold_num}")
    
    # Load test data
    test_file = f"./rep_{rep_num}/test_data/test_fold_{fold_num}.csv"
    if not os.path.exists(test_file):
        print(f"[Warning] Test file not found: {test_file}")
        return None
    
    test_data = pd.read_csv(test_file, dtype=str)
    
    # Remove orig_label_lc column if it exists
    if 'orig_label_lc' in test_data.columns:
        test_data = test_data.drop('orig_label_lc', axis=1)
    
    # Load NB model
    model_data = load_nb_model(rep_num, percentage, fold_num)
    if model_data is None:
        return None
    
    # Compute probabilities for the given iactions
    probability_result = compute_probability_for_iaction(model_data, test_data)
    if probability_result is None:
        return None
    
    # Create results dataframe
    results = test_data.copy()
    
    # Add probability of the given iaction (single column)
    results['probability'] = probability_result['probabilities']
    
    # DEBUG: Print probabilities for each iaction
    print(f"Probabilities for fold {fold_num}:")
    for iaction in sorted(results['iaction'].unique()):
        prob = results[results['iaction'] == iaction]['probability'].iloc[0]
        print(f"  P({iaction} | state) = {prob:.6f}")
    
    # All rows in the same test file belong to the same group
    # The fold number indicates the group_id
    results['group_id'] = fold_num
    
    # Add elapsed_time (placeholder for NB)
    results['elapsed_time'] = 0.0
    
    # Calculate ranking within the group (HIGHER probability = better rank for NB)
    # In Naive Bayes, we want interventions with HIGHER P(iaction | state)
    results['ranking'] = results['probability'].rank(method='dense', ascending=False).astype(int)
    
    # DEBUG: Print ranking information
    print(f"Ranking for group {fold_num}:")
    for _, row in results.iterrows():
        print(f"  {row['iaction']}: prob={row['probability']:.6f}, rank={row['ranking']}")
    
    # Add frequency column
    results = add_frequency_column(results, frequency_dict)
    
    # Calculate potential_crash_before and after intervention
    results = calculate_crash_potential(results, no_crashes_df)
    
    # Drop the original 'action' column after using it for crash calculation
    if 'action' in results.columns:
        results = results.drop('action', axis=1)
    
    # Initialize best_intervention column
    results['best_intervention'] = ''
    
    # Mark best interventions (ranking == 1) with '*'
    results.loc[results['ranking'] == 1, 'best_intervention'] = '*'
    
    # DEBUG: Print final best interventions
    best_interventions = results[results['best_intervention'] == '*']
    print(f"Best interventions for group {fold_num}:")
    for _, row in best_interventions.iterrows():
        print(f"  {row['iaction']} (prob={row['probability']:.6f})")
    
    return results

def calculate_crash_potential(df, no_crashes_df):
    """Calculate potential_crash_before and after intervention"""
    # Identify columns to exclude for state-key
    exclude_cols = set(['iaction', 'probability', 'elapsed_time', 'group_id', 'ranking', 'best_intervention', 'frequency'])
    for to_ex in ['latent_collision', 'labeled_lc', 'orig_label_lc']:
        if to_ex in df.columns:
            exclude_cols.add(to_ex)

    state_cols = [c for c in df.columns if c not in exclude_cols]

    # Create lookup set for no_crashes keys
    no_crashes_cols = [c for c in state_cols if c in no_crashes_df.columns]
    if len(no_crashes_cols) == 0:
        print("[Warning] No common columns with no_crashes data")
        df['potential_crash_before_intervention'] = 'True'
        df['potential_crash_after_intervention'] = 'True'
        return df

    no_crashes_key = make_key_from_df(no_crashes_df[no_crashes_cols], no_crashes_cols)
    no_crashes_set = set(no_crashes_key.values)

    # Before intervention - use the ORIGINAL action to determine crash potential
    before_key_match = make_key_from_df(df[no_crashes_cols], no_crashes_cols)
    df['potential_crash_before_intervention'] = np.where(before_key_match.isin(no_crashes_set), 'False', 'True')

    # After intervention (given iaction)
    df_after_keys = df[no_crashes_cols].copy()
    if 'action' in df_after_keys.columns and 'iaction' in df.columns:
        # Replace original action with iaction for after intervention
        df_after_keys['action'] = df['iaction'].astype(str)
    after_key = make_key_from_df(df_after_keys, no_crashes_cols)
    df['potential_crash_after_intervention'] = np.where(after_key.isin(no_crashes_set), 'False', 'True')

    return df

def make_key_from_df(df, cols):
    """Return a Series of string keys by concatenating values in cols (all cast to str)."""
    return df[cols].astype(str).agg("_".join, axis=1)

def load_frequency_data():
    """Load and prepare frequency data from complete_DB_discrete.csv"""
    print("Loading frequency data from:", COMPLETE_DB_PATH)
    if not os.path.exists(COMPLETE_DB_PATH):
        print(f"[Warning] Complete DB file not found at {COMPLETE_DB_PATH}. Frequency column will be NA.")
        return None
    
    freq_df = pd.read_csv(COMPLETE_DB_PATH, dtype=str)
    
    # Create frequency mapping for action-state pairs
    state_cols = ['curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W']
    available_state_cols = [col for col in state_cols if col in freq_df.columns]
    
    if 'action' not in freq_df.columns:
        print("[Warning] 'action' column not found in frequency data. Cannot compute frequencies.")
        return None
    
    if not available_state_cols:
        print("[Warning] No state columns found in frequency data. Cannot compute frequencies.")
        return None
    
    print(f"Using state columns for frequency: {available_state_cols}")
    
    # Create key for frequency lookup (action + state)
    freq_df['freq_key'] = make_key_from_df(freq_df[['action'] + available_state_cols], ['action'] + available_state_cols)
    
    # Count frequencies
    freq_counts = freq_df.groupby('freq_key').size().reset_index(name='frequency')
    
    return freq_counts.set_index('freq_key')['frequency'].to_dict()

def add_frequency_column(df, frequency_dict):
    """Add frequency column to dataframe based on action-state pairs"""
    if frequency_dict is None:
        df['frequency'] = 'NA'
        return df
    
    state_cols = ['curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W']
    available_state_cols = [col for col in state_cols if col in df.columns]
    
    # Use iaction (given intervention action) for frequency calculation
    if 'iaction' not in df.columns or not available_state_cols:
        df['frequency'] = 'NA'
        return df
    
    # Create frequency key for each row using iaction (intervention action)
    df['freq_key'] = make_key_from_df(df[['iaction'] + available_state_cols], ['iaction'] + available_state_cols)
    
    # Map frequencies
    df['frequency'] = df['freq_key'].map(
        lambda x: str(frequency_dict.get(x, 0)) if pd.notna(x) else '0'
    )
    
    # Remove temporary column
    df.drop('freq_key', axis=1, inplace=True, errors='ignore')
    
    return df

def save_nb_results(rep_num, percentage, all_results):
    """Save NB results in the same format as twin networks"""
    if not all_results:
        print(f"[Warning] No results to save for rep {rep_num}, percentage {percentage}")
        return
    
    # Combine all fold results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Create output directory
    nb_dir = os.path.join(f"rep_{rep_num}", percentage, "NB")
    os.makedirs(nb_dir, exist_ok=True)
    
    # Save files in the same format as twin networks
    input_csv = os.path.join(nb_dir, "twin_networks_results.csv")
    bi_csv = os.path.join(nb_dir, "best_interventions.csv")
    ds_csv = os.path.join(nb_dir, "data_sorted.csv")
    ct_txt = os.path.join(nb_dir, "contingency_table.txt")
    ct_atom_txt = os.path.join(nb_dir, "contingency_table_atomized.txt")
    
    # Save main results (equivalent to twin_networks_results.csv)
    combined_results.to_csv(input_csv, index=False)
    print(f"Saved NB results to {input_csv}")
    
    # For best_interventions, select only rows with best_intervention == '*'
    best_int_save = combined_results[combined_results['best_intervention'] == '*'].copy()
    to_drop = [c for c in ['group_id', 'ranking', 'elapsed_time', 'best_intervention'] if c in best_int_save.columns]
    best_int_save = best_int_save.drop(columns=to_drop, errors='ignore')
    best_int_save.to_csv(bi_csv, index=False)
    
    # data_sorted sorted by group_id and ranking
    combined_results_sorted = combined_results.sort_values(['group_id', 'ranking']).reset_index(drop=True)
    combined_results_sorted.to_csv(ds_csv, index=False)
    
    # Generate contingency tables
    generate_contingency_tables(combined_results_sorted, ct_txt, ct_atom_txt, rep_num, percentage)
    
    return nb_dir

def generate_contingency_tables(df, ct_txt, ct_atom_txt, rep_num, percentage):
    """Generate contingency tables using existing logic"""
    # This would contain the extensive contingency table generation code from your original script
    # For brevity, I'm showing a simplified version - you should copy your full implementation here
    
    with open(ct_txt, "w") as out:
        out.write("=== NAIVE BAYES RESULTS ===\n")
        out.write(f"Rep: {rep_num}, Percentage: {percentage}\n\n")
        
        # Basic counts
        total_cases = len(df)
        crash_before = (df['potential_crash_before_intervention'] == 'True').sum()
        no_crash_before = (df['potential_crash_before_intervention'] == 'False').sum()
        
        out.write(f"Total test cases: {total_cases}\n")
        out.write(f"Crash before intervention: {crash_before}\n")
        out.write(f"No crash before intervention: {no_crash_before}\n\n")
        
        # Transition counts for best interventions only (ranking == 1)
        best_interventions = df[df['ranking'] == 1]
        
        true_to_true = ((best_interventions['potential_crash_before_intervention'] == 'True') & 
                       (best_interventions['potential_crash_after_intervention'] == 'True')).sum()
        true_to_false = ((best_interventions['potential_crash_before_intervention'] == 'True') & 
                        (best_interventions['potential_crash_after_intervention'] == 'False')).sum()
        false_to_true = ((best_interventions['potential_crash_before_intervention'] == 'False') & 
                        (best_interventions['potential_crash_after_intervention'] == 'True')).sum()
        false_to_false = ((best_interventions['potential_crash_before_intervention'] == 'False') & 
                         (best_interventions['potential_crash_after_intervention'] == 'False')).sum()
        
        out.write("Transition Matrix (Best Interventions Only):\n")
        out.write(f"True -> True: {true_to_true}\n")
        out.write(f"True -> False: {true_to_false}\n")
        out.write(f"False -> True: {false_to_true}\n")
        out.write(f"False -> False: {false_to_false}\n")
        
        # Success rates
        total_crashes_before = (best_interventions['potential_crash_before_intervention'] == 'True').sum()
        crashes_prevented = true_to_false
        if total_crashes_before > 0:
            prevention_rate = (crashes_prevented / total_crashes_before) * 100
            out.write(f"\nCrash prevention rate: {prevention_rate:.2f}% ({crashes_prevented}/{total_crashes_before})\n")
    
    # Create atomized version (simplified)
    with open(ct_atom_txt, "w") as out:
        out.write("=== NAIVE BAYES RESULTS (ATOMIZED) ===\n")
        out.write(f"Rep: {rep_num}, Percentage: {percentage}\n\n")
        
        # Action distribution for best interventions
        best_interventions = df[df['ranking'] == 1]
        action_counts = best_interventions['iaction'].value_counts()
        out.write("Best intervention action distribution:\n")
        for action, count in action_counts.items():
            out.write(f"{action}: {count}\n")
    
    print(f"Generated contingency tables for NB results")

def process_nb_rep_perc(rep_num, perc, no_crashes_df):
    """Process all NB folds for a given repetition and percentage"""
    print(f"\nProcessing NB rep: {rep_num} percentage: {perc}")
    
    # Load frequency data
    frequency_dict = load_frequency_data()
    
    # Find how many test folds exist
    test_dir = f"./rep_{rep_num}/test_data"
    if not os.path.exists(test_dir):
        print(f"[Warning] Test directory not found: {test_dir}")
        return
    
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_fold_') and f.endswith('.csv')]
    num_folds = len(test_files)
    
    if num_folds == 0:
        print(f"[Warning] No test files found in {test_dir}")
        return
    
    all_results = []
    
    # Process each fold
    for fold_num in range(1, num_folds + 1):
        try:
            results = process_nb_fold(rep_num, perc, fold_num, no_crashes_df, frequency_dict)
            if results is not None:
                all_results.append(results)
                print(f"  Fold {fold_num}/{num_folds} processed successfully")
            else:
                print(f"  Fold {fold_num}/{num_folds} failed")
        except Exception as e:
            print(f"  [Error] processing fold {fold_num}: {e}")
    
    # Save combined results
    if all_results:
        save_nb_results(rep_num, perc, all_results)
        print(f"Successfully processed {len(all_results)} folds for rep {rep_num}, percentage {perc}")
    else:
        print(f"No successful results for rep {rep_num}, percentage {perc}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 summarize_nb_results.py <num_reps> <percentages_comma_separated>")
        print("Example: python3 summarize_nb_results.py 5 01,25,50,75,90")
        sys.exit(1)

    num_reps = int(sys.argv[1])
    percentages = sys.argv[2].split(",")

    # Load no_crashes
    if not os.path.exists(NO_CRASHES_PATH):
        print(f"[Error] no_crashes file not found at {NO_CRASHES_PATH}")
        sys.exit(1)
    no_crashes_df = pd.read_csv(NO_CRASHES_PATH, dtype=str)

    # Process NB results
    for rep in range(1, num_reps + 1):
        for perc in percentages:
            try:
                process_nb_rep_perc(rep, perc, no_crashes_df)
            except Exception as e:
                print(f"[Error] processing NB rep {rep} perc {perc}: {e}", flush=True)

    print("\nAll NB processing done.")

if __name__ == "__main__":
    main()
