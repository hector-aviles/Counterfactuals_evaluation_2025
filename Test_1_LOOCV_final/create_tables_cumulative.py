import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

# Configuration
percentages = ['01', '25', '50', '75', '90']
base_dirs = [
    './Test_1_LOOCV_rep_1',
    './Test_1_LOOCV_rep_2', 
    './Test_1_LOOCV_rep_3',
    './Test_1_LOOCV_rep_4',
    './Test_1_LOOCV_rep_5'
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
        # Build the correct file path based on percentage
        file_name = "contingency_table.csv"
        dir_path = os.path.join(base_dir, p, 'cBNs')
        file_path = os.path.join(dir_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Split into sections
                sections = re.split(r'=== (CRASH SECTION|NO CRASH SECTION|FINAL SUMMARY) ===', content)
                sections = [s.strip() for s in sections if s.strip()]
                
                # Process CRASH SECTION
                if len(sections) > 1 and 'CRASH SECTION' in sections[0]:
                    crash_content = sections[1]
                    
                    # 1. Process ties
                    for j in range(6):
                        pattern = re.compile(rf'Number of groups with exactly\s+{j+1}\s+best interventions:\s+(\d+)')
                        match = pattern.search(crash_content)
                        if match:
                            crash_ties_data[p][j] += safe_int_convert(match.group(1))
                    
                    # 2. Process actions and safety per action
                    action_block_match = re.search(r'Number of actions selected for each intervention type:(.*?)Total safe/unsafe actions:', crash_content, re.DOTALL)
                    if action_block_match:
                        action_block = action_block_match.group(1)
                        for line in action_block.split('\n'):
                            if ':' in line and any(action in line for action in valid_actions):
                                for action in valid_actions:
                                    if action in line:
                                        # Extract numbers using simpler pattern
                                        numbers = re.findall(r'\d+', line)
                                        if len(numbers) >= 3:
                                            count = safe_int_convert(numbers[0])
                                            safe_count = safe_int_convert(numbers[1])
                                            crash_actions_data[p][action] += count
                                            crash_action_safety_data[p][action] += safe_count
                    
                    # 3. Process safety
                    safe_match = re.search(r'True -> False:\s+(\d+)', crash_content)
                    if safe_match:
                        crash_safety_data[p]['Safe'] += safe_int_convert(safe_match.group(1))
                    
                    unsafe_match = re.search(r'True -> True:\s+(\d+)', crash_content)
                    if unsafe_match:
                        crash_safety_data[p]['Unsafe'] += safe_int_convert(unsafe_match.group(1))
                
                # Process NO CRASH SECTION
                if len(sections) > 3 and 'NO CRASH SECTION' in sections[2]:
                    no_crash_content = sections[3]
                    
                    # 1. Process ties
                    for j in range(6):
                        pattern = re.compile(rf'Number of groups with exactly\s+{j+1}\s+best interventions:\s+(\d+)')
                        match = pattern.search(no_crash_content)
                        if match:
                            no_crash_ties_data[p][j] += safe_int_convert(match.group(1))
                    
                    # 2. Process actions and safety per action
                    action_block_match = re.search(r'Number of actions selected for each intervention type:(.*?)Total safe/unsafe actions:', no_crash_content, re.DOTALL)
                    if action_block_match:
                        action_block = action_block_match.group(1)
                        for line in action_block.split('\n'):
                            if ':' in line and any(action in line for action in valid_actions):
                                for action in valid_actions:
                                    if action in line:
                                        # Extract numbers using simpler pattern
                                        numbers = re.findall(r'\d+', line)
                                        if len(numbers) >= 3:
                                            count = safe_int_convert(numbers[0])
                                            safe_count = safe_int_convert(numbers[1])
                                            no_crash_actions_data[p][action] += count
                                            no_crash_action_safety_data[p][action] += safe_count
                    
                    # 3. Process transitions
                    safe_match = re.search(r'False -> False:\s+(\d+)', no_crash_content)
                    if safe_match:
                        no_crash_transition_counts[p]['False_False'] += safe_int_convert(safe_match.group(1))
                    
                    unsafe_match = re.search(r'False -> True:\s+(\d+)', no_crash_content)
                    if unsafe_match:
                        no_crash_transition_counts[p]['False_True'] += safe_int_convert(unsafe_match.group(1))
                
                # Process FINAL SUMMARY
                if len(sections) > 5 and 'FINAL SUMMARY' in sections[4]:
                    summary_content = sections[5]
                    
                    for transition in ['True_True', 'True_False', 'False_True', 'False_False']:
                        if transition == 'True_True':
                            pattern = r'Crash before \(True\) and after intervention \(True\):\s+(\d+)'
                        elif transition == 'True_False':
                            pattern = r'Crash before \(True\) and after intervention \(False\):\s+(\d+)'
                        elif transition == 'False_True':
                            pattern = r'Crash before \(False\) and after intervention \(True\):\s+(\d+)'
                        elif transition == 'False_False':
                            pattern = r'Crash before \(False\) and after intervention \(False\):\s+(\d+)'
                        
                        match = re.search(pattern, summary_content)
                        if match:
                            final_summary[transition] += safe_int_convert(match.group(1))
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
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
        # File 1: Detailed results by section
        with open('cumulative_detailed_results.tex', 'w') as f:
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

\begin{table}[h]
\centering
\caption{Cumulative Safe and Unsafe Outcomes by Data Training Percentage (Crash Scenarios)}
\begin{tabular}{l|ccccc|c}
\toprule
\multirow{2}{*}{Outcome} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
Safe (True $\rightarrow$ False) & """)
            f.write(" & ".join(str(cumulative_crash_safety[p]['Safe']) for p in percentages))
            f.write(f" & {sum(cumulative_crash_safety[p]['Safe'] for p in percentages)} \\\\ \n")
            f.write(r"""Unsafe (True $\rightarrow$ True) & """)
            f.write(" & ".join(str(cumulative_crash_safety[p]['Unsafe']) for p in percentages))
            f.write(f" & {sum(cumulative_crash_safety[p]['Unsafe'] for p in percentages)} \\\\ \n")
            f.write(r"""\midrule
Total & """)
            f.write(" & ".join(str(cumulative_crash_safety[p]['Safe'] + cumulative_crash_safety[p]['Unsafe']) for p in percentages))
            f.write(f""" & {sum(cumulative_crash_safety[p]['Safe'] + cumulative_crash_safety[p]['Unsafe'] for p in percentages)} \\\\ \n
\\bottomrule
\end{{tabular}}
\end{{table}}
""")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Cumulative Number of Ties by Data Training Percentage (Crash Scenarios)}
\begin{tabular}{c|ccccc|c}
\toprule
\multirow{2}{*}{\# of ties (1st Place)} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
""")
            for i in range(6):
                row_total = sum(cumulative_crash_ties[p][i] for p in percentages)
                f.write(f"{i+1} & {cumulative_crash_ties['01'][i]} & {cumulative_crash_ties['25'][i]} & {cumulative_crash_ties['50'][i]} & {cumulative_crash_ties['75'][i]} & {cumulative_crash_ties['90'][i]} & {row_total} \\\\ \n")
            column_totals = [sum(cumulative_crash_ties[p]) for p in percentages]
            grand_total = sum(column_totals)
            f.write(f"""\midrule
Total & {' & '.join(map(str, column_totals))} & {grand_total} \\
\\bottomrule
\end{{tabular}}
\end{{table}}
""")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Cumulative Number of Actions Selected by Data Training Percentage (Crash Scenarios)}
\begin{tabular}{l|ccccc|c|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} & \multirow{2}{*}{Safe (\%)} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & & \\
\midrule
""")
            for action in valid_actions:
                action_tex = action.replace('_', '\\_')
                counts = [cumulative_crash_actions[p][action] for p in percentages]
                total = sum(counts)
                safe_counts = [cumulative_crash_action_safety[p][action] for p in percentages]
                safe_total = sum(safe_counts)
                safe_pct = safe_total / total * 100 if total > 0 else 0
                f.write(f"\\texttt{{{action_tex}}} & {' & '.join(map(str, counts))} & {total} & {safe_pct:.1f}\\% \\\\ \n")
            
            # Totals row
            total_counts = [sum(cumulative_crash_actions[p].values()) for p in percentages]
            total_total = sum(total_counts)
            safe_total_counts = [sum(cumulative_crash_action_safety[p].values()) for p in percentages]
            safe_total_total = sum(safe_total_counts)
            safe_total_pct = safe_total_total / total_total * 100 if total_total > 0 else 0
            f.write(f"""\midrule
\\textbf{{Total}} & {' & '.join(map(str, total_counts))} & {total_total} & {safe_total_pct:.1f}\\% \\\\ \n""")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}

\pagebreak
\section{No Crashes}

\begin{table}[h]
\centering
\caption{Cumulative Safe and Unsafe Outcomes by Data Training Percentage (No Crash Scenarios)}
\begin{tabular}{l|ccccc|c}
\toprule
\multirow{2}{*}{Outcome} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
""")
            f.write("Safe (False $\\rightarrow$ False) & ")
            f.write(" & ".join(str(cumulative_no_crash_transitions[p]['False_False']) for p in percentages))
            f.write(f" & {sum(cumulative_no_crash_transitions[p]['False_False'] for p in percentages)} \\\\ \n")
            f.write(r"""Unsafe (False $\rightarrow$ True) & """)
            f.write(" & ".join(str(cumulative_no_crash_transitions[p]['False_True']) for p in percentages))
            f.write(f" & {sum(cumulative_no_crash_transitions[p]['False_True'] for p in percentages)} \\\\ \n")
            f.write(r"""\midrule
Total & """)
            f.write(" & ".join(str(cumulative_no_crash_transitions[p]['False_False'] + cumulative_no_crash_transitions[p]['False_True']) for p in percentages))
            f.write(f""" & {sum(cumulative_no_crash_transitions[p]['False_False'] + cumulative_no_crash_transitions[p]['False_True'] for p in percentages)} \\\\ \n
\\bottomrule
\end{{tabular}}
\end{{table}}
""")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Cumulative Number of Ties by Data Training Percentage (No Crash Scenarios)}
\begin{tabular}{c|ccccc|c}
\toprule
\multirow{2}{*}{\# of ties (1st Place)} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
""")
            for i in range(6):
                row_total = sum(cumulative_no_crash_ties[p][i] for p in percentages)
                f.write(f"{i+1} & {cumulative_no_crash_ties['01'][i]} & {cumulative_no_crash_ties['25'][i]} & {cumulative_no_crash_ties['50'][i]} & {cumulative_no_crash_ties['75'][i]} & {cumulative_no_crash_ties['90'][i]} & {row_total} \\\\ \n")
            column_totals = [sum(cumulative_no_crash_ties[p]) for p in percentages]
            grand_total = sum(column_totals)
            f.write(f"""\midrule
Total & {' & '.join(map(str, column_totals))} & {grand_total} \\
\\bottomrule
\end{{tabular}}
\end{{table}}
""")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Cumulative Number of Actions Selected by Data Training Percentage (No Crash Scenarios)}
\begin{tabular}{l|ccccc|c|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} & \multirow{2}{*}{Safe (\%)} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & & \\
\midrule
""")
            for action in valid_actions:
                action_tex = action.replace('_', '\\_')
                counts = [cumulative_no_crash_actions[p][action] for p in percentages]
                total = sum(counts)
                safe_counts = [cumulative_no_crash_action_safety[p][action] for p in percentages]
                safe_total = sum(safe_counts)
                safe_pct = safe_total / total * 100 if total > 0 else 0
                f.write(f"\\texttt{{{action_tex}}} & {' & '.join(map(str, counts))} & {total} & {safe_pct:.1f}\\% \\\\ \n")
            
            # Totals row
            total_counts = [sum(cumulative_no_crash_actions[p].values()) for p in percentages]
            total_total = sum(total_counts)
            safe_total_counts = [sum(cumulative_no_crash_action_safety[p].values()) for p in percentages]
            safe_total_total = sum(safe_total_counts)
            safe_total_pct = safe_total_total / total_total * 100 if total_total > 0 else 0
            f.write(f"""\midrule
\\textbf{{Total}} & {' & '.join(map(str, total_counts))} & {total_total} & {safe_total_pct:.1f}\\% \\\\ \n""")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}
\end{document}
""")

        # File 2: Summary results with accumulated data
        with open('cumulative_summary_results.tex', 'w') as f:
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
            f.write(f"Safe Interventions & {sum(cumulative_crash_safety[p]['Safe'] for p in percentages)} & {sum(cumulative_no_crash_transitions[p]['False_False'] for p in percentages)} \\\\ \n")
            f.write(r"""Unsafe Interventions & """)
            f.write(f"{sum(cumulative_crash_safety[p]['Unsafe'] for p in percentages)} & {sum(cumulative_no_crash_transitions[p]['False_True'] for p in percentages)} \\\\ \n")
            f.write(r"""\midrule
Total & """)
            f.write(f"{sum(cumulative_crash_safety[p]['Safe'] + cumulative_crash_safety[p]['Unsafe'] for p in percentages)} & {sum(cumulative_no_crash_transitions[p]['False_False'] + cumulative_no_crash_transitions[p]['False_True'] for p in percentages)} \\\\ \n")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}
""")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Cumulative Accumulated Number of Ties Across All Percentages}
\begin{tabular}{c|cc}
\toprule
\multirow{2}{*}{\# of ties (1st Place)} & \multicolumn{2}{c|}{Scenario} \\
\cmidrule{2-3}
 & Crash Scenarios & No Crash Scenarios \\
\midrule
""")
            for i in range(6):
                crash_total = sum(cumulative_crash_ties[p][i] for p in percentages)
                no_crash_total = sum(cumulative_no_crash_ties[p][i] for p in percentages)
                f.write(f"{i+1} & {crash_total} & {no_crash_total} \\\\ \n")
            crash_grand_total = sum(sum(cumulative_crash_ties[p]) for p in percentages)
            no_crash_grand_total = sum(sum(cumulative_no_crash_ties[p]) for p in percentages)
            f.write(f"""\midrule
Total & {crash_grand_total} & {no_crash_grand_total} \\
\\bottomrule
\end{{tabular}}
\end{{table}}
""")

            f.write(r"""
\begin{table}[h]
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
                action_tex = action.replace('_', '\\_')
                crash_total = sum(cumulative_crash_actions[p][action] for p in percentages)
                crash_safe = sum(cumulative_crash_action_safety[p][action] for p in percentages)
                crash_pct = crash_safe / crash_total * 100 if crash_total > 0 else 0
                no_crash_total = sum(cumulative_no_crash_actions[p][action] for p in percentages)
                no_crash_safe = sum(cumulative_no_crash_action_safety[p][action] for p in percentages)
                no_crash_pct = no_crash_safe / no_crash_total * 100 if no_crash_total > 0 else 0
                f.write(f"\\texttt{{{action_tex}}} & {crash_total} ({crash_pct:.1f}\\% safe) & {no_crash_total} ({no_crash_pct:.1f}\\% safe) & {crash_total + no_crash_total} \\\\ \n")
            
            # Totals row
            crash_total_all = sum(sum(cumulative_crash_actions[p].values()) for p in percentages)
            crash_safe_all = sum(sum(cumulative_crash_action_safety[p].values()) for p in percentages)
            crash_pct_all = crash_safe_all / crash_total_all * 100 if crash_total_all > 0 else 0
            no_crash_total_all = sum(sum(cumulative_no_crash_actions[p].values()) for p in percentages)
            no_crash_safe_all = sum(sum(cumulative_no_crash_action_safety[p].values()) for p in percentages)
            no_crash_pct_all = no_crash_safe_all / no_crash_total_all * 100 if no_crash_total_all > 0 else 0
            f.write(f"""\midrule
\\textbf{{Total}} & {crash_total_all} ({crash_pct_all:.1f}\\% safe) & {no_crash_total_all} ({no_crash_pct_all:.1f}\\% safe) & {crash_total_all + no_crash_total_all} \\\\ \n""")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}

\section{Final Summary}

\begin{table}[h]
\centering
\caption{Cumulative Final Summary of Random Selection Outcomes}
\begin{tabular}{lc}
\toprule
Outcome & Count \\
\midrule
""")
            f.write(f"Crash unsafe (True $\\rightarrow$ True) & {cumulative_final_summary['True_True']} \\\\ \n")
            f.write(r"""Crash safe (True $\rightarrow$ False) & """)
            f.write(f"{cumulative_final_summary['True_False']} \\\\ \n")
            f.write(r"""No crash safe (False $\rightarrow$ False) & """)
            f.write(f"{cumulative_final_summary['False_False']} \\\\ \n")
            f.write(r"""No crash unsafe (False $\rightarrow$ True) & """)
            f.write(f"{cumulative_final_summary['False_True']} \\\\ \n")
            f.write(r"""\midrule
Total & """)
            f.write(f"{sum(cumulative_final_summary.values())} \\\\ \n")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}

\end{document}
""")

        print("Cumulative LaTeX files generated successfully:")
        print("- cumulative_detailed_results.tex")
        print("- cumulative_summary_results.tex")

    except Exception as e:
        print(f"Error generating cumulative LaTeX files: {str(e)}")
        raise

# Execute the function
if __name__ == "__main__":
    generate_cumulative_latex()
