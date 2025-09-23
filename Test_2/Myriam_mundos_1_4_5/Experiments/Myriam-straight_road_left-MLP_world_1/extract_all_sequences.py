import pandas as pd
import numpy as np
import os
import random

random.seed(123)

# Load the CSV file
file_path = 'preprocessed.csv'
df = pd.read_csv(file_path, low_memory=False)

# Extract the important columns - including num_trial and success if they exist
columns_to_extract = ['iteration', 'av_pos.x', 'av_pos.y', 'car3_pos.x', 'car3_pos.y']
if 'num_trial' in df.columns:
    columns_to_extract.append('num_trial')
if 'success' in df.columns:
    columns_to_extract.append('success')

df = df[columns_to_extract]

# Group the data by sequences (each sequence starts with iteration == 1)
sequences = []
current_sequence = []
for _, row in df.iterrows():
    if row['iteration'] == 1 and current_sequence:  # Start of a new sequence
        sequences.append(current_sequence)
        current_sequence = []
    current_sequence.append(row)
if current_sequence:  # Add the last sequence
    sequences.append(current_sequence)

# Convert sequences to DataFrames
sequence_dfs = [pd.DataFrame(seq) for seq in sequences]

# Initialize counters for successful and failed trials
successful_trials = []
failed_trials = []

# Process each sequence to check success status
for i, seq_df in enumerate(sequence_dfs):
    # Check if 'success' column exists in this sequence
    if 'success' in seq_df.columns:
        # Get the last success value (assuming it's constant throughout the sequence)
        success_value = seq_df['success'].iloc[-1]
        
        # Get the trial number if available
        trial_num = seq_df['num_trial'].iloc[0] if 'num_trial' in seq_df.columns else i+1
        
        if success_value:
            successful_trials.append(trial_num)
        else:
            failed_trials.append(trial_num)

# Print results
print("\nSuccess/Failure Report:")
if successful_trials:
    print(f"Successful trials (num_trial): {', '.join(map(str, successful_trials))}")
else:
    print("No successful trials found")

if failed_trials:
    print(f"Failed trials (num_trial): {', '.join(map(str, failed_trials))}")
else:
    print("No failed trials found")

# Extract the last value of av_pos.x for each sequence
last_av_pos_x_values = [seq_df['av_pos.x'].iloc[-1] for seq_df in sequence_dfs]

# Sum the last values of av_pos.x
total_meters = sum(last_av_pos_x_values)

# Convert the total from meters to kilometers
total_kilometers = total_meters / 1000

# Display the result
print(f"\nTotal distance traveled: {total_kilometers:.2f} kilometers")

# Save the selected sequences to disk
output_dir = 'all_sequences'
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

for i, seq_df in enumerate(sequence_dfs):
    output_file = os.path.join(output_dir, f'sequence_{i + 1}.csv')
    seq_df.to_csv(output_file, index=False)
    print(f'Saved {output_file}')

print(f"\nSaved {len(sequence_dfs)} sequences to '{output_dir}'.")
