import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
percentages = ['01', '25', '50', '75', '90']
base_dirs = [
    '../rep_1',
    '../rep_2', 
    '../rep_3',
    '../rep_4',
    '../rep_5'
]

def safe_int_convert(value_str):
    try:
        return int(value_str.strip())
    except:
        return 0

# Initialize data structures
crash_safe_percentages = {p: [] for p in percentages}  # List of percentages for each repetition
no_crash_safe_percentages = {p: [] for p in percentages}

# Keep track of missing files
missing_files = []

# Process all repetitions
for base_dir in base_dirs:
    print(f"Processing directory: {base_dir}")
    
    for p in percentages:
        # Build the correct file path
        file_name = "contingency_table.txt"
        dir_path = os.path.join(base_dir, p, 'cBNs')
        file_path = os.path.join(dir_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            missing_files.append(file_path)
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Split into main sections (only the top-level sections)
                sections = re.split(r'=== (CRASH SECTION|NO CRASH SECTION) ===', content)
                sections = [s.strip() for s in sections if s.strip()]
                
                # Process CRASH SECTION - only get the first occurrence from the main section
                if len(sections) > 1 and 'CRASH SECTION' in sections[0]:
                    crash_content = sections[1]
                    main_crash_section = crash_content.split('Detailed transition matrices')[0]
                    
                    safe_match = re.search(r'True -> False:\s+(\d+)', main_crash_section)
                    unsafe_match = re.search(r'True -> True:\s+(\d+)', main_crash_section)
                    
                    if safe_match and unsafe_match:
                        safe_count = safe_int_convert(safe_match.group(1))
                        unsafe_count = safe_int_convert(unsafe_match.group(1))
                        total = safe_count + unsafe_count
                        
                        if total > 0:
                            safe_percentage = (safe_count / total) * 100
                            crash_safe_percentages[p].append(safe_percentage)
                
                # Process NO CRASH SECTION - only get the first occurrence from the main section
                if len(sections) > 3 and 'NO CRASH SECTION' in sections[2]:
                    no_crash_content = sections[3]
                    main_no_crash_section = no_crash_content.split('Detailed transition matrices')[0]
                    
                    safe_match = re.search(r'False -> False:\s+(\d+)', main_no_crash_section)
                    unsafe_match = re.search(r'False -> True:\s+(\d+)', main_no_crash_section)
                    
                    if safe_match and unsafe_match:
                        safe_count = safe_int_convert(safe_match.group(1))
                        unsafe_count = safe_int_convert(unsafe_match.group(1))
                        total = safe_count + unsafe_count
                        
                        if total > 0:
                            safe_percentage = (safe_count / total) * 100
                            no_crash_safe_percentages[p].append(safe_percentage)
                            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

# Report missing files
if missing_files:
    print("\n=== MISSING FILES SUMMARY ===")
    print(f"Total missing files: {len(missing_files)}")
    for f in missing_files:
        print(f"  {f}")
else:
    print("\nAll files were found.")

# Calculate statistics
x_values = [int(p) for p in percentages]

crash_means = []
crash_stds = []
no_crash_means = []
no_crash_stds = []

for p in percentages:
    # Crash scenarios
    if crash_safe_percentages[p]:
        crash_means.append(np.mean(crash_safe_percentages[p]))
        crash_stds.append(np.std(crash_safe_percentages[p]))
    else:
        crash_means.append(0)
        crash_stds.append(0)
    
    # No crash scenarios
    if no_crash_safe_percentages[p]:
        no_crash_means.append(np.mean(no_crash_safe_percentages[p]))
        no_crash_stds.append(np.std(no_crash_safe_percentages[p]))
    else:
        no_crash_means.append(0)
        no_crash_stds.append(0)

# Create the plot with error bars
plt.figure(figsize=(12, 8))

# Plot crash scenarios with error bars
plt.errorbar(x_values, crash_means, yerr=crash_stds, fmt='o-', 
             linewidth=2, markersize=8, capsize=6, capthick=2,
             label='Driving scenarios originally labeled as unsafe', color='blue', alpha=0.8)

# Plot no crash scenarios with error bars
plt.errorbar(x_values, no_crash_means, yerr=no_crash_stds, fmt='s-', 
             linewidth=2, markersize=8, capsize=6, capthick=2,
             label='Driving scenarios originally labeled as safe', color='red', alpha=0.8)

# Custom annotation positions
annotation_offsets_crash = [(5, 10), (0, 10), (0, 10), (0, 10), (0, 10)]
annotation_offsets_no_crash = [(5, -32), (0, -20), (0, -15), (0, -15), (0, -15)]

for i, (x, y, std) in enumerate(zip(x_values, crash_means, crash_stds)):
    offset_x, offset_y = annotation_offsets_crash[i]
    if i == 0:
        plt.annotate(f'{y:.1f}% ± {std:.1f}', (x, y), xytext=(offset_x, offset_y), textcoords='offset points',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    else:
        plt.annotate(f'{y:.1f}% ± {std:.1f}', (x, y), xytext=(offset_x, offset_y), textcoords='offset points',
                     fontsize=12, ha='center', va='bottom',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

for i, (x, y, std) in enumerate(zip(x_values, no_crash_means, no_crash_stds)):
    offset_x, offset_y = annotation_offsets_no_crash[i]
    if i == 0:
        plt.annotate(f'{y:.1f}% ± {std:.1f}', (x, y), xytext=(offset_x, offset_y), textcoords='offset points',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    else:
        plt.annotate(f'{y:.1f}% ± {std:.1f}', (x, y), xytext=(offset_x, offset_y), textcoords='offset points',
                     fontsize=12, ha='center', va='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

# Customize the plot
plt.xlabel('Data Training Percentage', fontsize=16)
plt.ylabel('Percentage of Safe Actions (%)', fontsize=16)
plt.title('Effect of Training Data Percentage on Safe Action Selection\n(Average ± Standard Deviation across 5 Repetitions)', 
          fontsize=18, pad=20)
plt.legend(fontsize=14, loc='lower right')
plt.xticks(x_values, [f'{p}%' for p in percentages], fontsize=14)
plt.xlim(0, 95)
plt.ylim(92, 102)
yticks = plt.yticks()[0]
yticks = [tick for tick in yticks if tick != 102]
plt.yticks(yticks, fontsize=14)
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.98, f'Total repetitions: {len(base_dirs)}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.tight_layout()
plt.savefig('safe_actions_percentage_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print("\nCrash Scenarios:")
for i, p in enumerate(percentages):
    if crash_safe_percentages[p]:
        print(f"  {p}%: {crash_means[i]:.1f}% ± {crash_stds[i]:.1f}% "
              f"(range: {min(crash_safe_percentages[p]):.1f}%-{max(crash_safe_percentages[p]):.1f}%)")
    else:
        print(f"  {p}%: No data")

print("\nNo Crash Scenarios:")
for i, p in enumerate(percentages):
    if no_crash_safe_percentages[p]:
        print(f"  {p}%: {no_crash_means[i]:.1f}% ± {no_crash_stds[i]:.1f}% "
              f"(range: {min(no_crash_safe_percentages[p]):.1f}%-{max(no_crash_safe_percentages[p]):.1f}%)")
    else:
        print(f"  {p}%: No data")

