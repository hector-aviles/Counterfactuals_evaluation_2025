import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib as mpl

# Directory containing the sequence files
input_directory = "./all_sequences"
output_directory = "./all_sequences_output"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Number of sequences
num_sequences = 30

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'cm'

# Loop through each sequence file
for i in range(1, num_sequences + 1):
    # Construct the input file path
    file_path = os.path.join(input_directory, f"sequence_{i}.csv")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping...")
        continue
    
    # Read the sequence from the CSV file
    df = pd.read_csv(file_path)
    
    # Filter data
    filtered_data = df.dropna()
    filtered_data = filtered_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Apply filters for both trajectories
    condition = (filtered_data['av_pos.y'].abs() < 4) & (filtered_data['av_pos.x'] < 2000)
    filtered_data = filtered_data[condition]
    
    # Get trajectory data
    av_x = filtered_data['av_pos.x'].values
    av_y = filtered_data['av_pos.y'].values
    car3_x = filtered_data['car3_pos.x'].values
    car3_y = filtered_data['car3_pos.y'].values
    
    # Remove zeros
    av_non_zero = np.logical_and(av_x != 0, av_y != 0)
    av_x = av_x[av_non_zero]
    av_y = av_y[av_non_zero]
    
    car3_non_zero = np.logical_and(car3_x != 0, car3_y != 0)
    car3_x = car3_x[car3_non_zero]
    car3_y = car3_y[car3_non_zero]
    
    # Create a new figure for this sequence
    plt.figure(figsize=(15, 6))
    plt.ylim(-4, 4)
    
    # Plot the trajectories
    plt.plot(av_x, av_y, marker='o', linestyle='-', color='red', 
             linewidth=0.5, markersize=0.5, label='AV Trajectory')
    plt.plot(car3_x, car3_y, marker='o', linestyle='-', color='darkgoldenrod', 
             linewidth=0.5, markersize=0.5, label='Car3 Trajectory')
    
    # Add a horizontal dashed line at y=0
    plt.axhline(y=0, color='gray', linestyle='dashed', linewidth=2, label='Lane marking')
    
    # Add labels and title
    plt.xlabel(r'$x$-coordinate (m)', fontsize=12)
    plt.ylabel(r'$y$-coordinate (m)', fontsize=12)
    plt.title(f'Trajectories for Sequence {i}')
    
    # Add legend
    plt.legend()
    
    # Save the plot as PDF
    output_path = os.path.join(output_directory, f"sequence_{i}.pdf")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved plot for sequence {i} to {output_path}")

print("All plots generated successfully.")
