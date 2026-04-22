import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

# Configuration
percentages = ['01', '50', '100']
base_dirs = [
    '../rep_1'
]
valid_actions = ['cruise', 'keep', 'change_to_left', 'change_to_right', 'swerve_left', 'swerve_right']

# Initialize data structures for all repetitions
all_crash_ties_data = []
all_no_crash_ties_data = []
all_crash_actions_data = []
all_no_crash_actions_data = []
all_crash_action_safety_data = []
all_no_crash_action_safety_data = []
all_crash_safety_data = []
all_no_crash_transition_counts = []
all_final_summary = []

# NEW: per-tie-level unsafe counts for crash and no-crash sections
all_crash_ties_unsafe_data = []
all_no_crash_ties_unsafe_data = []

def safe_int_convert(value_str):
    try:
        return int(value_str.strip())
    except:
        return 0

# Track missing files
missing_files = []

# Process all repetitions
for base_dir in base_dirs:
    print(f"Processing directory: {base_dir}")
    
    # Initialize data structures for this repetition
    crash_ties_data = {p: [0]*6 for p in percentages}
    no_crash_ties_data = {p: [0]*6 for p in percentages}
    crash_actions_data = {p: {a: 0 for a in valid_actions} for p in percentages}
    no_crash_actions_data = {p: {a: 0 for a in valid_actions} for p in percentages}
    crash_action_safety_data = {p: {a: 0 for a in valid_actions} for p in percentages}
    no_crash_action_safety_data = {p: {a: 0 for a in valid_actions} for p in percentages}
    crash_safety_data = {p: {'Safe': 0, 'Unsafe': 0} for p in percentages}
    no_crash_transition_counts = {p: {'False_True': 0, 'False_False': 0} for p in percentages}
    final_summary = {'True_True': 0, 'True_False': 0, 'False_True': 0, 'False_False': 0}

    # NEW: per-tie-level unsafe counts [tie_level_1..6] for crash and no-crash sections
    crash_ties_unsafe_data = {p: [0]*6 for p in percentages}
    no_crash_ties_unsafe_data = {p: [0]*6 for p in percentages}

    # Process files for this repetition
    for p in percentages:
        file_name = "contingency_table.txt"
        dir_path = os.path.join(base_dir, p, 'cBNs')
        file_path = os.path.join(dir_path, file_name)

        # Check file existence
        if not os.path.exists(file_path):
            print(f"Warning: Missing file {file_path}")
            missing_files.append(file_path)
            continue

        try:
            with open(file_path, 'r') as f:
                content = f.read()
                sections = re.split(r'=== (CRASH SECTION|NO CRASH SECTION|FINAL SUMMARY) ===', content)
                sections = [s.strip() for s in sections if s.strip()]

                # ----------- CRASH SECTION -----------
                if len(sections) > 1 and 'CRASH SECTION' in sections[0]:
                    crash_content = sections[1]
                    for j in range(6):
                        match = re.search(rf'Number of groups with exactly\s+{j+1}\s+best interventions:\s+(\d+)', crash_content)
                        if match:
                            crash_ties_data[p][j] += safe_int_convert(match.group(1))

                    action_block_match = re.search(
                        r'Number of actions selected for each intervention type:(.*?)Total safe/unsafe actions:',
                        crash_content, re.DOTALL)
                    if action_block_match:
                        action_block = action_block_match.group(1)
                        for line in action_block.split('\n'):
                            if ':' in line:
                                for action in valid_actions:
                                    if action in line:
                                        nums = re.findall(r'\d+', line)
                                        if len(nums) >= 3:
                                            crash_actions_data[p][action] += safe_int_convert(nums[0])
                                            crash_action_safety_data[p][action] += safe_int_convert(nums[1])

                    safe_match = re.search(r'True -> False:\s+(\d+)', crash_content)
                    if safe_match:
                        crash_safety_data[p]['Safe'] += safe_int_convert(safe_match.group(1))
                    unsafe_match = re.search(r'True -> True:\s+(\d+)', crash_content)
                    if unsafe_match:
                        crash_safety_data[p]['Unsafe'] += safe_int_convert(unsafe_match.group(1))

                    # NEW: Parse per-tie-level unsafe counts from
                    # "Detailed transition matrices by number of ties (CRASH SECTION)"
                    detailed_block_match = re.search(
                        r'Detailed transition matrices by number of ties \(CRASH SECTION\):(.*?)(?:=== NO CRASH SECTION ===|$)',
                        crash_content, re.DOTALL)
                    if detailed_block_match:
                        detailed_block = detailed_block_match.group(1)
                        for j in range(1, 7):
                            # Find the block for exactly j tied interventions
                            tie_block_match = re.search(
                                rf'For groups with {j} tied best interventions:.*?True -> True:\s+(\d+)',
                                detailed_block, re.DOTALL)
                            if tie_block_match:
                                crash_ties_unsafe_data[p][j-1] += safe_int_convert(tie_block_match.group(1))

                # ----------- NO CRASH SECTION -----------
                if len(sections) > 3 and 'NO CRASH SECTION' in sections[2]:
                    no_crash_content = sections[3]
                    for j in range(6):
                        match = re.search(rf'Number of groups with exactly\s+{j+1}\s+best interventions:\s+(\d+)', no_crash_content)
                        if match:
                            no_crash_ties_data[p][j] += safe_int_convert(match.group(1))

                    action_block_match = re.search(
                        r'Number of actions selected for each intervention type:(.*?)Total safe/unsafe actions:',
                        no_crash_content, re.DOTALL)
                    if action_block_match:
                        action_block = action_block_match.group(1)
                        for line in action_block.split('\n'):
                            if ':' in line:
                                for action in valid_actions:
                                    if action in line:
                                        nums = re.findall(r'\d+', line)
                                        if len(nums) >= 3:
                                            no_crash_actions_data[p][action] += safe_int_convert(nums[0])
                                            no_crash_action_safety_data[p][action] += safe_int_convert(nums[1])

                    safe_match = re.search(r'False -> False:\s+(\d+)', no_crash_content)
                    if safe_match:
                        no_crash_transition_counts[p]['False_False'] += safe_int_convert(safe_match.group(1))
                    unsafe_match = re.search(r'False -> True:\s+(\d+)', no_crash_content)
                    if unsafe_match:
                        no_crash_transition_counts[p]['False_True'] += safe_int_convert(unsafe_match.group(1))

                    # NEW: Parse per-tie-level unsafe counts from
                    # "Detailed transition matrices by number of ties (NO CRASH SECTION)"
                    # Unsafe in no-crash section = False -> True
                    nc_detailed_block_match = re.search(
                        r'Detailed transition matrices by number of ties \(NO CRASH SECTION\):(.*?)(?:=== FINAL SUMMARY ===|$)',
                        no_crash_content, re.DOTALL)
                    if nc_detailed_block_match:
                        nc_detailed_block = nc_detailed_block_match.group(1)
                        for j in range(1, 7):
                            tie_block_match = re.search(
                                rf'For groups with {j} tied best interventions:.*?False -> True:\s+(\d+)',
                                nc_detailed_block, re.DOTALL)
                            if tie_block_match:
                                no_crash_ties_unsafe_data[p][j-1] += safe_int_convert(tie_block_match.group(1))

                # ----------- FINAL SUMMARY -----------
                if len(sections) > 5 and 'FINAL SUMMARY' in sections[4]:
                    summary_content = sections[5]
                    patterns = {
                        'True_True': r'Crash before \(True\) and after intervention \(True\):\s+(\d+)',
                        'True_False': r'Crash before \(True\) and after intervention \(False\):\s+(\d+)',
                        'False_True': r'Crash before \(False\) and after intervention \(True\):\s+(\d+)',
                        'False_False': r'Crash before \(False\) and after intervention \(False\):\s+(\d+)',
                    }
                    for key, pattern in patterns.items():
                        match = re.search(pattern, summary_content)
                        if match:
                            final_summary[key] += safe_int_convert(match.group(1))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Store data for this repetition
    all_crash_ties_data.append(crash_ties_data)
    all_no_crash_ties_data.append(no_crash_ties_data)
    all_crash_actions_data.append(crash_actions_data)
    all_no_crash_actions_data.append(no_crash_actions_data)
    all_crash_action_safety_data.append(crash_action_safety_data)
    all_no_crash_action_safety_data.append(no_crash_action_safety_data)
    all_crash_safety_data.append(crash_safety_data)
    all_no_crash_transition_counts.append(no_crash_transition_counts)
    all_final_summary.append(final_summary)
    all_crash_ties_unsafe_data.append(crash_ties_unsafe_data)  # NEW
    all_no_crash_ties_unsafe_data.append(no_crash_ties_unsafe_data)  # NEW

# Final report of missing files
if missing_files:
    print("\nSummary: Missing Files Report")
    print(f"Total missing files: {len(missing_files)}")
    for f in missing_files:
        print(f"  - {f}")
else:
    print("\nAll expected files were found.")

# Calculate cumulative results
def calculate_cumulative(data_list):
    cumulative = {}
    for p in percentages:
        if isinstance(data_list[0][p], dict):
            cumulative[p] = {}
            for key in data_list[0][p].keys():
                cumulative[p][key] = sum(rep[p][key] for rep in data_list)
        elif isinstance(data_list[0][p], list):
            cumulative[p] = [0] * len(data_list[0][p])
            for i in range(len(data_list[0][p])):
                cumulative[p][i] = sum(rep[p][i] for rep in data_list)
    return cumulative

# Calculate cumulative results
cumulative_crash_ties = calculate_cumulative(all_crash_ties_data)
cumulative_no_crash_ties = calculate_cumulative(all_no_crash_ties_data)
cumulative_crash_safety = calculate_cumulative(all_crash_safety_data)
cumulative_no_crash_transitions = calculate_cumulative(all_no_crash_transition_counts)
cumulative_crash_actions = calculate_cumulative(all_crash_actions_data)
cumulative_no_crash_actions = calculate_cumulative(all_no_crash_actions_data)
cumulative_crash_action_safety = calculate_cumulative(all_crash_action_safety_data)
cumulative_no_crash_action_safety = calculate_cumulative(all_no_crash_action_safety_data)
cumulative_crash_ties_unsafe = calculate_cumulative(all_crash_ties_unsafe_data)  # NEW
cumulative_no_crash_ties_unsafe = calculate_cumulative(all_no_crash_ties_unsafe_data)  # NEW

cumulative_final_summary = {}
for key in all_final_summary[0].keys():
    cumulative_final_summary[key] = sum(rep[key] for rep in all_final_summary)

# Generate LaTeX files with cumulative results
def generate_cumulative_latex():
    try:
        # Helper to write a LaTeX row safely (ends with \\ newline)
        def write_row(fobj, row_text):
            fobj.write(row_text + " \\\\ \n")

        # Open detailed results file
        with open('cumulative_detailed_results_with_unsafe.tex', 'w') as f:
            f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\title{Cumulative Results - TPLP 2025}
\author{havilesa}
\date{July 2025}
\begin{document}
\maketitle

\section{Crashes}
""")

            # --- Safe/Unsafe table (Crash) ---
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Safe and Unsafe Outcomes by Data Training Percentage (Crash Scenarios)}
\begin{tabular}{l|ccc|c}
\toprule
\multirow{2}{*}{Outcome} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-4}
 & 01 & 50 & 100 & \\
\midrule
""")
            # Safe row
            safe_vals = [str(cumulative_crash_safety[p]['Safe']) for p in percentages]
            f.write("Safe (True $\\rightarrow$ False) & " + " & ".join(safe_vals))
            f.write(" & " + str(sum(int(x) for x in safe_vals)) + " \\\\ \n")
            # Unsafe row
            unsafe_vals = [str(cumulative_crash_safety[p]['Unsafe']) for p in percentages]
            f.write("Unsafe (True $\\rightarrow$ True) & " + " & ".join(unsafe_vals))
            f.write(" & " + str(sum(int(x) for x in unsafe_vals)) + " \\\\ \n")
            # Total row
            totals = [str(cumulative_crash_safety[p]['Safe'] + cumulative_crash_safety[p]['Unsafe']) for p in percentages]
            f.write(r"\midrule" + "\n")
            f.write("Total & " + " & ".join(totals))
            f.write(" & " + str(sum(int(x) for x in totals)) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # --- Ties table (Crash) --- MODIFIED to include unsafe counts per tie level
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Number of Ties by Data Training Percentage (Crash Scenarios).
Each cell shows tied actions / query groups (unsafe actions in parentheses).}
\begin{tabular}{c|ccc|c}
\toprule
\multirow{2}{*}{\# of optimal actions} & \multicolumn{3}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-4}
 & 01 & 50 & 100 & \\
\midrule
""")
            for i in range(6):
                row_parts = []
                for p in percentages:
                    n_groups = cumulative_crash_ties[p][i]
                    n_tied_actions = n_groups * (i + 1)  # total tied actions = groups * tie_level
                    n_unsafe = cumulative_crash_ties_unsafe[p][i]
                    if n_groups == 0:
                        row_parts.append("0/0")
                    else:
                        row_parts.append(f"{n_tied_actions}/{n_groups} ({n_unsafe})")
                # Total column: sum across percentages
                total_groups = sum(cumulative_crash_ties[p][i] for p in percentages)
                total_tied = total_groups * (i + 1)
                total_unsafe = sum(cumulative_crash_ties_unsafe[p][i] for p in percentages)
                if total_groups == 0:
                    total_str = "0/0"
                else:
                    total_str = f"{total_tied}/{total_groups} ({total_unsafe})"
                f.write(str(i+1) + " & " + " & ".join(row_parts) + " & " + total_str + " \\\\ \n")

            # Totals row (groups only, unsafe total)
            col_totals_groups = [str(sum(cumulative_crash_ties[p])) for p in percentages]
            col_totals_unsafe = [str(sum(cumulative_crash_ties_unsafe[p])) for p in percentages]
            grand_groups = sum(int(x) for x in col_totals_groups)
            grand_unsafe = sum(int(x) for x in col_totals_unsafe)
            f.write(r"\midrule" + "\n")
            # For total row just show group counts and total unsafe
            f.write("Total groups & " + " & ".join(
                [f"{col_totals_groups[j]} ({col_totals_unsafe[j]})" for j in range(len(percentages))]
            ) + " & " + f"{grand_groups} ({grand_unsafe})" + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # --- Actions table (Crash) ---
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Number of Actions Selected by Data Training Percentage (Crash Scenarios)}
\begin{tabular}{l|ccc|c|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} & \multirow{2}{*}{Safe (\%)} \\
\cmidrule{2-4}
 & 01 & 50 & 100 & & \\
\midrule
""")
            for action in valid_actions:
                counts = [cumulative_crash_actions[p][action] for p in percentages]
                counts_str = [str(x) for x in counts]
                total = sum(counts)
                safe_counts = [cumulative_crash_action_safety[p][action] for p in percentages]
                safe_total = sum(safe_counts)
                safe_pct = (safe_total / total * 100) if total > 0 else 0.0
                f.write("\\texttt{" + action.replace('_', '\\_') + "} & " + " & ".join(counts_str) + " & " + str(total) + " & " + f"{safe_pct:.1f}\\% \\\\ \n")
            # Totals row
            total_counts = [sum(cumulative_crash_actions[p].values()) for p in percentages]
            total_counts_str = [str(x) for x in total_counts]
            total_total = sum(total_counts)
            safe_total_counts = [sum(cumulative_crash_action_safety[p].values()) for p in percentages]
            safe_total_total = sum(safe_total_counts)
            safe_total_pct = (safe_total_total / total_total * 100) if total_total > 0 else 0.0
            f.write(r"\midrule" + "\n")
            f.write("\\textbf{Total} & " + " & ".join(total_counts_str) + " & " + str(total_total) + " & " + f"{safe_total_pct:.1f}\\% \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # --- No Crashes section ---
            f.write(r"\pagebreak" + "\n" + r"\section{No Crashes}" + "\n\n")

            # Safe/Unsafe No Crash table
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Safe and Unsafe Outcomes by Data Training Percentage (No Crash Scenarios)}
\begin{tabular}{l|ccc|c}
\toprule
\multirow{2}{*}{Outcome} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-4}
 & 01 & 50  & 100 & \\
\midrule
""")
            no_crash_safe = [str(cumulative_no_crash_transitions[p]['False_False']) for p in percentages]
            no_crash_unsafe = [str(cumulative_no_crash_transitions[p]['False_True']) for p in percentages]
            f.write("Safe (False $\\rightarrow$ False) & " + " & ".join(no_crash_safe) + " & " + str(sum(int(x) for x in no_crash_safe)) + " \\\\ \n")
            f.write("Unsafe (False $\\rightarrow$ True) & " + " & ".join(no_crash_unsafe) + " & " + str(sum(int(x) for x in no_crash_unsafe)) + " \\\\ \n")
            f.write(r"\midrule" + "\n")
            totals_nc = [str(int(a) + int(b)) for a, b in zip(no_crash_safe, no_crash_unsafe)]
            f.write("Total & " + " & ".join(totals_nc) + " & " + str(sum(int(x) for x in totals_nc)) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Ties (No Crash) - now includes unsafe counts per tie level
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Number of Ties by Data Training Percentage (No Crash Scenarios).
Each cell shows tied actions / query groups (unsafe actions in parentheses).}
\begin{tabular}{c|ccc|c}
\toprule
\multirow{2}{*}{\# of optimal actions} & \multicolumn{3}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-4}
 & 01 & 50 & 100 & \\
\midrule
""")
            for i in range(6):
                row_parts = []
                for p in percentages:
                    n_groups = cumulative_no_crash_ties[p][i]
                    n_tied_actions = n_groups * (i + 1)
                    n_unsafe = cumulative_no_crash_ties_unsafe[p][i]
                    if n_groups == 0:
                        row_parts.append("0/0")
                    else:
                        row_parts.append(f"{n_tied_actions}/{n_groups} ({n_unsafe})")
                total_groups = sum(cumulative_no_crash_ties[p][i] for p in percentages)
                total_tied = total_groups * (i + 1)
                total_unsafe = sum(cumulative_no_crash_ties_unsafe[p][i] for p in percentages)
                if total_groups == 0:
                    total_str = "0/0"
                else:
                    total_str = f"{total_tied}/{total_groups} ({total_unsafe})"
                f.write(str(i+1) + " & " + " & ".join(row_parts) + " & " + total_str + " \\\\ \n")
            col_totals_nc_groups = [str(sum(cumulative_no_crash_ties[p])) for p in percentages]
            col_totals_nc_unsafe = [str(sum(cumulative_no_crash_ties_unsafe[p])) for p in percentages]
            grand_nc_groups = sum(int(x) for x in col_totals_nc_groups)
            grand_nc_unsafe = sum(int(x) for x in col_totals_nc_unsafe)
            f.write(r"\midrule" + "\n")
            f.write("Total groups & " + " & ".join(
                [f"{col_totals_nc_groups[j]} ({col_totals_nc_unsafe[j]})" for j in range(len(percentages))]
            ) + " & " + f"{grand_nc_groups} ({grand_nc_unsafe})" + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Actions (No Crash)
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Number of Actions Selected by Data Training Percentage (No Crash Scenarios)}
\begin{tabular}{l|ccc|c|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} & \multirow{2}{*}{Safe (\%)} \\
\cmidrule{2-4}
 & 01 & 50  & 100 & & \\
\midrule
""")
            for action in valid_actions:
                counts = [cumulative_no_crash_actions[p][action] for p in percentages]
                counts_str = [str(x) for x in counts]
                total = sum(counts)
                safe_counts = [cumulative_no_crash_action_safety[p][action] for p in percentages]
                safe_total = sum(safe_counts)
                safe_pct = (safe_total / total * 100) if total > 0 else 0.0
                f.write("\\texttt{" + action.replace('_', '\\_') + "} & " + " & ".join(counts_str) + " & " + str(total) + " & " + f"{safe_pct:.1f}\\% \\\\ \n")
            total_counts_nc = [str(sum(cumulative_no_crash_actions[p].values())) for p in percentages]
            total_total_nc = sum(int(x) for x in total_counts_nc)
            safe_total_counts_nc = [str(sum(cumulative_no_crash_action_safety[p].values())) for p in percentages]
            safe_total_total_nc = sum(int(x) for x in safe_total_counts_nc)
            safe_total_pct_nc = (safe_total_total_nc / total_total_nc * 100) if total_total_nc > 0 else 0.0
            f.write(r"\midrule" + "\n")
            f.write("\\textbf{Total} & " + " & ".join(total_counts_nc) + " & " + str(total_total_nc) + " & " + f"{safe_total_pct_nc:.1f}\\% \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # --- Grand Total section ---
            f.write(r"\pagebreak" + "\n" + r"\section{Grand Total (Crash + No Crash)}" + "\n\n")

            # Grand Total Safe/Unsafe table
            f.write(r"""\begin{table}[h]
\centering
\caption{Grand Total Safe and Unsafe Outcomes by Data Training Percentage (Crash + No Crash)}
\begin{tabular}{l|ccc|c}
\toprule
\multirow{2}{*}{Outcome} & \multicolumn{3}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-4}
 & 01 & 50 & 100 & \\
\midrule
""")
            gt_safe_vals = [str(
                cumulative_crash_safety[p]['Safe'] +
                cumulative_no_crash_transitions[p]['False_False']
            ) for p in percentages]
            gt_unsafe_vals = [str(
                cumulative_crash_safety[p]['Unsafe'] +
                cumulative_no_crash_transitions[p]['False_True']
            ) for p in percentages]
            gt_total_vals = [str(int(s) + int(u)) for s, u in zip(gt_safe_vals, gt_unsafe_vals)]
            f.write("Safe & " + " & ".join(gt_safe_vals) + " & " + str(sum(int(x) for x in gt_safe_vals)) + " \\\\ \n")
            f.write("Unsafe & " + " & ".join(gt_unsafe_vals) + " & " + str(sum(int(x) for x in gt_unsafe_vals)) + " \\\\ \n")
            f.write(r"\midrule" + "\n")
            f.write("Total & " + " & ".join(gt_total_vals) + " & " + str(sum(int(x) for x in gt_total_vals)) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Grand Total Ties table
            f.write(r"""\begin{table}[h]
\centering
\caption{Grand Total Number of Ties by Data Training Percentage (Crash + No Crash).
Each cell shows tied actions / query groups (unsafe actions in parentheses).}
\begin{tabular}{c|ccc|c}
\toprule
\multirow{2}{*}{\# of optimal actions} & \multicolumn{3}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-4}
 & 01 & 50 & 100 & \\
\midrule
""")
            for i in range(6):
                row_parts = []
                for p in percentages:
                    n_groups = cumulative_crash_ties[p][i] + cumulative_no_crash_ties[p][i]
                    n_tied_actions = n_groups * (i + 1)
                    n_unsafe = cumulative_crash_ties_unsafe[p][i] + cumulative_no_crash_ties_unsafe[p][i]
                    if n_groups == 0:
                        row_parts.append("0/0")
                    else:
                        row_parts.append(f"{n_tied_actions}/{n_groups} ({n_unsafe})")
                total_groups = sum(
                    cumulative_crash_ties[p][i] + cumulative_no_crash_ties[p][i]
                    for p in percentages)
                total_tied = total_groups * (i + 1)
                total_unsafe = sum(
                    cumulative_crash_ties_unsafe[p][i] + cumulative_no_crash_ties_unsafe[p][i]
                    for p in percentages)
                total_str = f"{total_tied}/{total_groups} ({total_unsafe})" if total_groups > 0 else "0/0"
                f.write(str(i+1) + " & " + " & ".join(row_parts) + " & " + total_str + " \\\\ \n")
            gt_col_groups = [str(sum(cumulative_crash_ties[p]) + sum(cumulative_no_crash_ties[p])) for p in percentages]
            gt_col_unsafe = [str(sum(cumulative_crash_ties_unsafe[p]) + sum(cumulative_no_crash_ties_unsafe[p])) for p in percentages]
            gt_grand_groups = sum(int(x) for x in gt_col_groups)
            gt_grand_unsafe = sum(int(x) for x in gt_col_unsafe)
            f.write(r"\midrule" + "\n")
            f.write("Total groups & " + " & ".join(
                [f"{gt_col_groups[j]} ({gt_col_unsafe[j]})" for j in range(len(percentages))]
            ) + " & " + f"{gt_grand_groups} ({gt_grand_unsafe})" + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Grand Total Actions table
            f.write(r"""\begin{table}[h]
\centering
\caption{Grand Total Number of Actions Selected by Data Training Percentage (Crash + No Crash)}
\begin{tabular}{l|ccc|c|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{3}{c|}{Data training percentage} & \multirow{2}{*}{Total} & \multirow{2}{*}{Safe (\%)} \\
\cmidrule{2-4}
 & 01 & 50 & 100 & & \\
\midrule
""")
            for action in valid_actions:
                gt_counts = [
                    cumulative_crash_actions[p][action] + cumulative_no_crash_actions[p][action]
                    for p in percentages]
                gt_safe_counts = [
                    cumulative_crash_action_safety[p][action] + cumulative_no_crash_action_safety[p][action]
                    for p in percentages]
                gt_total = sum(gt_counts)
                gt_safe_total = sum(gt_safe_counts)
                gt_safe_pct = (gt_safe_total / gt_total * 100) if gt_total > 0 else 0.0
                f.write("\\texttt{" + action.replace('_', '\\_') + "} & " +
                        " & ".join(str(x) for x in gt_counts) + " & " +
                        str(gt_total) + " & " + f"{gt_safe_pct:.1f}\\% \\\\ \n")
            gt_total_counts = [
                sum(cumulative_crash_actions[p].values()) + sum(cumulative_no_crash_actions[p].values())
                for p in percentages]
            gt_safe_total_counts = [
                sum(cumulative_crash_action_safety[p].values()) + sum(cumulative_no_crash_action_safety[p].values())
                for p in percentages]
            gt_grand_total = sum(gt_total_counts)
            gt_grand_safe = sum(gt_safe_total_counts)
            gt_grand_pct = (gt_grand_safe / gt_grand_total * 100) if gt_grand_total > 0 else 0.0
            f.write(r"\midrule" + "\n")
            f.write("\\textbf{Total} & " + " & ".join(str(x) for x in gt_total_counts) +
                    " & " + str(gt_grand_total) + " & " + f"{gt_grand_pct:.1f}\\% \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # End of document
            f.write(r"\end{document}" + "\n")

        # --- Summary file ---
        with open('cumulative_summary_results_with_unsafe.tex', 'w') as f:
            f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\title{Cumulative Summary Results - TPLP 2025}
\author{havilesa}
\date{July 2025}
\begin{document}
\maketitle

\begin{table}[h]
\centering
\caption{Cumulative Accumulated Safe and Unsafe Outcomes Across All Percentages}
\begin{tabular}{l|cc}
\toprule
Category & Crash Scenarios & No Crash Scenarios \\
\midrule
""")
            f.write("Safe Interventions & " + str(sum(cumulative_crash_safety[p]['Safe'] for p in percentages)) + " & " + str(sum(cumulative_no_crash_transitions[p]['False_False'] for p in percentages)) + " \\\\ \n")
            f.write("Unsafe Interventions & " + str(sum(cumulative_crash_safety[p]['Unsafe'] for p in percentages)) + " & " + str(sum(cumulative_no_crash_transitions[p]['False_True'] for p in percentages)) + " \\\\ \n")
            f.write(r"\midrule" + "\n")
            f.write("Total & " + str(sum((cumulative_crash_safety[p]['Safe'] + cumulative_crash_safety[p]['Unsafe']) for p in percentages)) + " & " + str(sum((cumulative_no_crash_transitions[p]['False_False'] + cumulative_no_crash_transitions[p]['False_True']) for p in percentages)) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Ties aggregated — MODIFIED to include unsafe per tie level
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Accumulated Number of Ties Across All Percentages.
Unsafe actions in parentheses (crash section only).}
\begin{tabular}{c|cc}
\toprule
\multirow{2}{*}{\# of ties (1st Place)} & \multicolumn{2}{c|}{Scenario} \\
\cmidrule{2-3}
 & Crash Scenarios & No Crash Scenarios \\
\midrule
""")
            for i in range(6):
                crash_total = sum(cumulative_crash_ties[p][i] for p in percentages)
                crash_unsafe_total = sum(cumulative_crash_ties_unsafe[p][i] for p in percentages)
                no_crash_total = sum(cumulative_no_crash_ties[p][i] for p in percentages)
                no_crash_unsafe_total = sum(cumulative_no_crash_ties_unsafe[p][i] for p in percentages)
                crash_str = f"{crash_total} ({crash_unsafe_total})" if crash_total > 0 else "0"
                no_crash_str = f"{no_crash_total} ({no_crash_unsafe_total})" if no_crash_total > 0 else "0"
                f.write(str(i+1) + " & " + crash_str + " & " + no_crash_str + " \\\\ \n")
            crash_grand_total = sum(sum(cumulative_crash_ties[p]) for p in percentages)
            crash_grand_unsafe = sum(sum(cumulative_crash_ties_unsafe[p]) for p in percentages)
            no_crash_grand_total = sum(sum(cumulative_no_crash_ties[p]) for p in percentages)
            no_crash_grand_unsafe = sum(sum(cumulative_no_crash_ties_unsafe[p]) for p in percentages)
            f.write(r"\midrule" + "\n")
            f.write("Total & " + f"{crash_grand_total} ({crash_grand_unsafe})" + " & " + f"{no_crash_grand_total} ({no_crash_grand_unsafe})" + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Actions aggregated
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Accumulated Number of Actions Across All Percentages}
\begin{tabular}{l|cc|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{2}{c|}{Scenario} & \multirow{2}{*}{Total} \\
\cmidrule{2-3}
 & Crash Scenarios & No Crash Scenarios \\
\midrule
""")
            for action in valid_actions:
                crash_total = sum(cumulative_crash_actions[p][action] for p in percentages)
                crash_safe = sum(cumulative_crash_action_safety[p][action] for p in percentages)
                crash_pct = (crash_safe / crash_total * 100) if crash_total > 0 else 0.0
                no_crash_total = sum(cumulative_no_crash_actions[p][action] for p in percentages)
                no_crash_safe = sum(cumulative_no_crash_action_safety[p][action] for p in percentages)
                no_crash_pct = (no_crash_safe / no_crash_total * 100) if no_crash_total > 0 else 0.0
                f.write("\\texttt{" + action.replace('_', '\\_') + "} & " + str(crash_total) + " (" + f"{crash_pct:.1f}\\% safe" + ") & " + str(no_crash_total) + " (" + f"{no_crash_pct:.1f}\\% safe" + ") & " + str(crash_total + no_crash_total) + " \\\\ \n")
            crash_total_all = sum(sum(cumulative_crash_actions[p].values()) for p in percentages)
            crash_safe_all = sum(sum(cumulative_crash_action_safety[p].values()) for p in percentages)
            crash_pct_all = (crash_safe_all / crash_total_all * 100) if crash_total_all > 0 else 0.0
            no_crash_total_all = sum(sum(cumulative_no_crash_actions[p].values()) for p in percentages)
            no_crash_safe_all = sum(sum(cumulative_no_crash_action_safety[p].values()) for p in percentages)
            no_crash_pct_all = (no_crash_safe_all / no_crash_total_all * 100) if no_crash_total_all > 0 else 0.0
            f.write(r"\midrule" + "\n")
            f.write("\\textbf{Total} & " + str(crash_total_all) + " (" + f"{crash_pct_all:.1f}\\% safe" + ") & " + str(no_crash_total_all) + " (" + f"{no_crash_pct_all:.1f}\\% safe" + ") & " + str(crash_total_all + no_crash_total_all) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Final summary table
            f.write(r"""\section{Final Summary}

\begin{table}[h]
\centering
\caption{Cumulative Final Summary of Random Selection Outcomes}
\begin{tabular}{lc}
\toprule
Outcome & Count \\
\midrule
""")
            f.write("Crash unsafe (True $\\rightarrow$ True) & " + str(cumulative_final_summary['True_True']) + " \\\\ \n")
            f.write("Crash safe (True $\\rightarrow$ False) & " + str(cumulative_final_summary['True_False']) + " \\\\ \n")
            f.write("No crash safe (False $\\rightarrow$ False) & " + str(cumulative_final_summary['False_False']) + " \\\\ \n")
            f.write("No crash unsafe (False $\\rightarrow$ True) & " + str(cumulative_final_summary['False_True']) + " \\\\ \n")
            f.write(r"\midrule" + "\n")
            f.write("Total & " + str(sum(cumulative_final_summary.values())) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # --- Grand Total section in summary ---
            f.write(r"\section{Grand Total (Crash + No Crash)}" + "\n\n")

            # Grand Total Safe/Unsafe
            gt_safe = sum(cumulative_crash_safety[p]['Safe'] + cumulative_no_crash_transitions[p]['False_False'] for p in percentages)
            gt_unsafe = sum(cumulative_crash_safety[p]['Unsafe'] + cumulative_no_crash_transitions[p]['False_True'] for p in percentages)
            gt_total = gt_safe + gt_unsafe
            f.write(r"""\begin{table}[h]
\centering
\caption{Grand Total Safe and Unsafe Outcomes (Crash + No Crash, All Percentages)}
\begin{tabular}{lc}
\toprule
Outcome & Count \\
\midrule
""")
            f.write("Safe & " + str(gt_safe) + " \\\\ \n")
            f.write("Unsafe & " + str(gt_unsafe) + " \\\\ \n")
            f.write(r"\midrule" + "\n")
            f.write("Total & " + str(gt_total) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Grand Total Ties
            f.write(r"""\begin{table}[h]
\centering
\caption{Grand Total Number of Ties Across All Percentages (Crash + No Crash).
Unsafe actions in parentheses.}
\begin{tabular}{c|c}
\toprule
\# of optimal actions & Total (unsafe) \\
\midrule
""")
            for i in range(6):
                gt_groups = sum(cumulative_crash_ties[p][i] + cumulative_no_crash_ties[p][i] for p in percentages)
                gt_unsafe_i = sum(cumulative_crash_ties_unsafe[p][i] + cumulative_no_crash_ties_unsafe[p][i] for p in percentages)
                gt_str = f"{gt_groups} ({gt_unsafe_i})" if gt_groups > 0 else "0"
                f.write(str(i+1) + " & " + gt_str + " \\\\ \n")
            gt_grand_groups = sum(sum(cumulative_crash_ties[p]) + sum(cumulative_no_crash_ties[p]) for p in percentages)
            gt_grand_unsafe_ties = sum(sum(cumulative_crash_ties_unsafe[p]) + sum(cumulative_no_crash_ties_unsafe[p]) for p in percentages)
            f.write(r"\midrule" + "\n")
            f.write("Total & " + f"{gt_grand_groups} ({gt_grand_unsafe_ties})" + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Grand Total Actions
            f.write(r"""\begin{table}[h]
\centering
\caption{Grand Total Number of Actions Selected Across All Percentages (Crash + No Crash)}
\begin{tabular}{l|c|c}
\toprule
Action & Total & Safe (\%) \\
\midrule
""")
            for action in valid_actions:
                gt_act_total = sum(
                    cumulative_crash_actions[p][action] + cumulative_no_crash_actions[p][action]
                    for p in percentages)
                gt_act_safe = sum(
                    cumulative_crash_action_safety[p][action] + cumulative_no_crash_action_safety[p][action]
                    for p in percentages)
                gt_act_pct = (gt_act_safe / gt_act_total * 100) if gt_act_total > 0 else 0.0
                f.write("\\texttt{" + action.replace('_', '\\_') + "} & " +
                        str(gt_act_total) + " & " + f"{gt_act_pct:.1f}\\% \\\\ \n")
            gt_all_total = sum(
                sum(cumulative_crash_actions[p].values()) + sum(cumulative_no_crash_actions[p].values())
                for p in percentages)
            gt_all_safe = sum(
                sum(cumulative_crash_action_safety[p].values()) + sum(cumulative_no_crash_action_safety[p].values())
                for p in percentages)
            gt_all_pct = (gt_all_safe / gt_all_total * 100) if gt_all_total > 0 else 0.0
            f.write(r"\midrule" + "\n")
            f.write("\\textbf{Total} & " + str(gt_all_total) + " & " + f"{gt_all_pct:.1f}\\% \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")

            f.write(r"\end{document}" + "\n")

        print("Cumulative LaTeX files generated successfully:")
        print("- cumulative_detailed_results_with_unsafe.tex")
        print("- cumulative_summary_results_with_unsafe.tex")

    except Exception as e:
        print(f"Error generating cumulative LaTeX files: {str(e)}")
        raise


# Execute the function
if __name__ == "__main__":
    generate_cumulative_latex()
