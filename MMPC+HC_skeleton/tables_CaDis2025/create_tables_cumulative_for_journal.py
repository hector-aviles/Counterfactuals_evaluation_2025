import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

# Configuration
percentages = ['01', '25', '50', '75', '90']
base_dirs = [
    '../rep_1',
    '../rep_2', 
    '../rep_3',
    '../rep_4',
    '../rep_5'
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
        with open('cumulative_detailed_results_journal.tex', 'w') as f:
            f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\title{Cumulative Results - TPLP 2025}
\author{havilesa}
\date{July 2025}
\begin{document}
\maketitle

\section{Overall Results}
""")

            # --- Combined Safe/Unsafe table ---
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Safe and Unsafe Outcomes by Data Training Percentage (All Scenarios)}
\begin{tabular}{l|ccccc|c}
\toprule
\multirow{2}{*}{Outcome} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
""")
            # Safe row (Crash + No Crash)
            safe_vals = [str(cumulative_crash_safety[p]['Safe'] + cumulative_no_crash_transitions[p]['False_False']) for p in percentages]
            f.write("Safe Interventions & " + " & ".join(safe_vals))
            f.write(" & " + str(sum(int(x) for x in safe_vals)) + " \\\\ \n")
            # Unsafe row (Crash + No Crash)
            unsafe_vals = [str(cumulative_crash_safety[p]['Unsafe'] + cumulative_no_crash_transitions[p]['False_True']) for p in percentages]
            f.write("Unsafe Interventions & " + " & ".join(unsafe_vals))
            f.write(" & " + str(sum(int(x) for x in unsafe_vals)) + " \\\\ \n")
            # Total row
            totals = [str(int(safe) + int(unsafe)) for safe, unsafe in zip(safe_vals, unsafe_vals)]
            f.write(r"\midrule" + "\n")
            f.write("Total & " + " & ".join(totals))
            f.write(" & " + str(sum(int(x) for x in totals)) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # --- Combined Ties table ---
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Number of Ties by Data Training Percentage (All Scenarios)}
\begin{tabular}{c|ccccc|c}
\toprule
\multirow{2}{*}{\# of ties (1st Place)} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
""")
            for i in range(6):
                row_vals = [str(cumulative_crash_ties[p][i] + cumulative_no_crash_ties[p][i]) for p in percentages]
                row_total = sum(int(v) for v in row_vals)
                # Write row with safe concatenation
                f.write(str(i+1) + " & " + " & ".join(row_vals) + " & " + str(row_total) + " \\\\ \n")
            column_totals = [str(sum(cumulative_crash_ties[p]) + sum(cumulative_no_crash_ties[p])) for p in percentages]
            grand_total = sum(int(x) for x in column_totals)
            f.write(r"\midrule" + "\n")
            f.write("Total & " + " & ".join(column_totals) + " & " + str(grand_total) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # --- Combined Actions table ---
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Number of Actions Selected by Data Training Percentage (All Scenarios)}
\begin{tabular}{l|ccccc|c|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} & \multirow{2}{*}{Safe (\%)} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & & \\
\midrule
""")
            for action in valid_actions:
                counts = [cumulative_crash_actions[p][action] + cumulative_no_crash_actions[p][action] for p in percentages]
                counts_str = [str(x) for x in counts]
                total = sum(counts)
                safe_counts = [cumulative_crash_action_safety[p][action] + cumulative_no_crash_action_safety[p][action] for p in percentages]
                safe_total = sum(safe_counts)
                safe_pct = (safe_total / total * 100) if total > 0 else 0.0
                f.write("\\texttt{" + action.replace('_', '\\_') + "} & " + " & ".join(counts_str) + " & " + str(total) + " & " + f"{safe_pct:.1f}\\% \\\\ \n")
            # Totals row
            total_counts = [sum(cumulative_crash_actions[p].values()) + sum(cumulative_no_crash_actions[p].values()) for p in percentages]
            total_counts_str = [str(x) for x in total_counts]
            total_total = sum(total_counts)
            safe_total_counts = [sum(cumulative_crash_action_safety[p].values()) + sum(cumulative_no_crash_action_safety[p].values()) for p in percentages]
            safe_total_total = sum(safe_total_counts)
            safe_total_pct = (safe_total_total / total_total * 100) if total_total > 0 else 0.0
            f.write(r"\midrule" + "\n")
            f.write("\\textbf{Total} & " + " & ".join(total_counts_str) + " & " + str(total_total) + " & " + f"{safe_total_pct:.1f}\\% \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # End of document
            f.write(r"\end{document}" + "\n")

        # --- Summary file ---
        with open('cumulative_summary_results_journal.tex', 'w') as f:
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
\caption{Cumulative Overall Safe and Unsafe Outcomes}
\begin{tabular}{lc}
\toprule
Category & Count \\
\midrule
""")
            total_safe = sum(cumulative_crash_safety[p]['Safe'] for p in percentages) + sum(cumulative_no_crash_transitions[p]['False_False'] for p in percentages)
            total_unsafe = sum(cumulative_crash_safety[p]['Unsafe'] for p in percentages) + sum(cumulative_no_crash_transitions[p]['False_True'] for p in percentages)
            f.write("Safe Interventions & " + str(total_safe) + " \\\\ \n")
            f.write("Unsafe Interventions & " + str(total_unsafe) + " \\\\ \n")
            f.write(r"\midrule" + "\n")
            f.write("Total & " + str(total_safe + total_unsafe) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Ties aggregated
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Overall Number of Ties}
\begin{tabular}{cc}
\toprule
\# of ties (1st Place) & Count \\
\midrule
""")
            for i in range(6):
                total_ties = sum(cumulative_crash_ties[p][i] for p in percentages) + sum(cumulative_no_crash_ties[p][i] for p in percentages)
                f.write(str(i+1) + " & " + str(total_ties) + " \\\\ \n")
            grand_total_ties = sum(sum(cumulative_crash_ties[p]) for p in percentages) + sum(sum(cumulative_no_crash_ties[p]) for p in percentages)
            f.write(r"\midrule" + "\n")
            f.write("Total & " + str(grand_total_ties) + " \\\\ \n")
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n")

            # Actions aggregated
            f.write(r"""\begin{table}[h]
\centering
\caption{Cumulative Overall Number of Actions}
\begin{tabular}{l|c|c}
\toprule
Action & Count & Safe (\%) \\
\midrule
""")
            for action in valid_actions:
                crash_total = sum(cumulative_crash_actions[p][action] for p in percentages)
                crash_safe = sum(cumulative_crash_action_safety[p][action] for p in percentages)
                no_crash_total = sum(cumulative_no_crash_actions[p][action] for p in percentages)
                no_crash_safe = sum(cumulative_no_crash_action_safety[p][action] for p in percentages)
                total = crash_total + no_crash_total
                safe_total = crash_safe + no_crash_safe
                safe_pct = (safe_total / total * 100) if total > 0 else 0.0
                f.write("\\texttt{" + action.replace('_', '\\_') + "} & " + str(total) + " & " + f"{safe_pct:.1f}\\% \\\\ \n")
            crash_total_all = sum(sum(cumulative_crash_actions[p].values()) for p in percentages)
            crash_safe_all = sum(sum(cumulative_crash_action_safety[p].values()) for p in percentages)
            no_crash_total_all = sum(sum(cumulative_no_crash_actions[p].values()) for p in percentages)
            no_crash_safe_all = sum(sum(cumulative_no_crash_action_safety[p].values()) for p in percentages)
            total_all = crash_total_all + no_crash_total_all
            safe_total_all = crash_safe_all + no_crash_safe_all
            safe_pct_all = (safe_total_all / total_all * 100) if total_all > 0 else 0.0
            f.write(r"\midrule" + "\n")
            f.write("\\textbf{Total} & " + str(total_all) + " & " + f"{safe_pct_all:.1f}\\% \\\\ \n")
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
            f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")

            f.write(r"\end{document}" + "\n")

        print("Cumulative LaTeX files generated successfully:")
        print("- cumulative_detailed_results_journal.tex")
        print("- cumulative_summary_results_journal.tex")

    except Exception as e:
        print(f"Error generating cumulative LaTeX files: {str(e)}")
        raise


# Execute the function
if __name__ == "__main__":
    generate_cumulative_latex()
