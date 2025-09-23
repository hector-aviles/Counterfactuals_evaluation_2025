import os
import re
from collections import defaultdict

# Configuration
percentages = ['01', '25', '50', '75', '90']
base_dir = '.'  # Current directory is Test_1_LOOCV_rep_1
valid_actions = ['cruise', 'keep', 'change_to_left', 'change_to_right', 'swerve_left', 'swerve_right']

# Initialize data structures
crash_ties_data = {p: [0]*6 for p in percentages}
no_crash_ties_data = {p: [0]*6 for p in percentages}
crash_actions_data = {p: {a: 0 for a in valid_actions} for p in percentages}
no_crash_actions_data = {p: {a: 0 for a in valid_actions} for p in percentages}
crash_action_safety_data = {p: {a: 0 for a in valid_actions} for p in percentages}
no_crash_action_safety_data = {p: {a: 0 for a in valid_actions} for p in percentages}
crash_safety_data = {p: {'Safe': 0, 'Unsafe': 0} for p in percentages}
no_crash_transition_counts = {p: {'False_True': 0, 'False_False': 0} for p in percentages}
final_summary = {'True_True': 0, 'True_False': 0, 'False_True': 0, 'False_False': 0}

def safe_int_convert(value_str):
    try:
        return int(value_str.strip())
    except:
        return 0

# Process files
for p in percentages:
    # Build the correct file path - files are in subdirectories like "01/cBNs/"
    dir_path = os.path.join(base_dir, p, 'cBNs')
    file_name = "contingency_table.csv"    
    file_path = os.path.join(dir_path, file_name)
    
    print(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print(f"Looking in directory: {os.path.abspath(dir_path)}")
        # List files in the directory to debug
        if os.path.exists(dir_path):
            print(f"Files in {dir_path}: {os.listdir(dir_path)}")
        continue
        
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            print(f"File size: {len(content)} characters")
            
            # Use the robust section parsing
            sections = re.split(r'=== (CRASH SECTION|NO CRASH SECTION|FINAL SUMMARY) ===', content)
            sections = [s.strip() for s in sections if s.strip()]
            
            print(f"Found {len(sections)} sections")
            
            # Process CRASH SECTION
            crash_content = None
            no_crash_content = None
            summary_content = None
            
            for i, section in enumerate(sections):
                if section == 'CRASH SECTION' and i + 1 < len(sections):
                    crash_content = sections[i + 1]
                elif section == 'NO CRASH SECTION' and i + 1 < len(sections):
                    no_crash_content = sections[i + 1]
                elif section == 'FINAL SUMMARY' and i + 1 < len(sections):
                    summary_content = sections[i + 1]
            
            # Process CRASH SECTION
            if crash_content:
                print(f"Processing CRASH SECTION for {p}")
                
                # 1. Process ties
                for j in range(6):
                    pattern = re.compile(rf'Number of groups with exactly\s+{j+1}\s+best interventions:\s+(\d+)')
                    match = pattern.search(crash_content)
                    if match:
                        crash_ties_data[p][j] = safe_int_convert(match.group(1))
                        print(f"Found {j+1} ties in crash: {match.group(1)}")
                    else:
                        print(f"No match found for {j+1} ties in crash section")
                
                # 2. Process actions
                action_lines = re.findall(r'(\w+(?:_\w+)*)\s*:\s*Selected\s+(\d+)\s*times,\s*Safe\s+(\d+)\s*times,\s*Unsafe\s+(\d+)\s*times', crash_content)
                for action, count_str, safe_str, unsafe_str in action_lines:
                    if action in valid_actions:
                        count = safe_int_convert(count_str)
                        safe_count = safe_int_convert(safe_str)
                        crash_actions_data[p][action] = count
                        crash_action_safety_data[p][action] = safe_count
                        print(f"Found action {action}: count={count}, safe={safe_count}")
                
                # 3. Process safety transitions
                safe_match = re.search(r'True -> False:\s+(\d+)', crash_content)
                if safe_match:
                    crash_safety_data[p]['Safe'] = safe_int_convert(safe_match.group(1))
                    print(f"Found Safe (True -> False): {safe_match.group(1)}")
                
                unsafe_match = re.search(r'True -> True:\s+(\d+)', crash_content)
                if unsafe_match:
                    crash_safety_data[p]['Unsafe'] = safe_int_convert(unsafe_match.group(1))
                    print(f"Found Unsafe (True -> True): {safe_match.group(1)}")
            
            # Process NO CRASH SECTION
            if no_crash_content:
                print(f"Processing NO CRASH SECTION for {p}")
                
                # 1. Process ties
                for j in range(6):
                    pattern = re.compile(rf'Number of groups with exactly\s+{j+1}\s+best interventions:\s+(\d+)')
                    match = pattern.search(no_crash_content)
                    if match:
                        no_crash_ties_data[p][j] = safe_int_convert(match.group(1))
                        print(f"Found {j+1} ties in no crash: {match.group(1)}")
                
                # 2. Process actions
                action_lines = re.findall(r'(\w+(?:_\w+)*)\s*:\s*Selected\s+(\d+)\s*times,\s*Safe\s+(\d+)\s*times,\s*Unsafe\s+(\d+)\s*times', no_crash_content)
                for action, count_str, safe_str, unsafe_str in action_lines:
                    if action in valid_actions:
                        count = safe_int_convert(count_str)
                        safe_count = safe_int_convert(safe_str)
                        no_crash_actions_data[p][action] = count
                        no_crash_action_safety_data[p][action] = safe_count
                        print(f"Found action {action}: count={count}, safe={safe_count}")
                
                # 3. Process transitions
                safe_match = re.search(r'False -> False:\s+(\d+)', no_crash_content)
                if safe_match:
                    no_crash_transition_counts[p]['False_False'] = safe_int_convert(safe_match.group(1))
                    print(f"Found Safe (False -> False): {safe_match.group(1)}")
                
                unsafe_match = re.search(r'False -> True:\s+(\d+)', no_crash_content)
                if unsafe_match:
                    no_crash_transition_counts[p]['False_True'] = safe_int_convert(safe_match.group(1))
                    print(f"Found Unsafe (False -> True): {safe_match.group(1)}")
            
            # Process FINAL SUMMARY
            if summary_content:
                print(f"Processing FINAL SUMMARY for {p}")
                
                for transition in ['True_True', 'True_False', 'False_True', 'False_False']:
                    before = 'True' if transition.startswith('True') else 'False'
                    after = 'True' if transition.endswith('True') else 'False'
                    pattern = re.compile(rf'Random selection - Crash before \({before}\) and after intervention \({after}\):\s+(\d+)')
                    match = pattern.search(summary_content)
                    if match:
                        final_summary[transition] += safe_int_convert(match.group(1))
                        print(f"Found Final Summary {transition}: {match.group(1)}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Print the parsed data to verify
print("\n=== PARSED DATA ===")
print("Crash ties data:", crash_ties_data)
print("No crash ties data:", no_crash_ties_data)
print("Crash actions data:", crash_actions_data)
print("No crash actions data:", no_crash_actions_data)
print("Crash safety data:", crash_safety_data)
print("No crash transition counts:", no_crash_transition_counts)
print("Final summary:", final_summary)

# Now let's test with a simple manual parse to see what's wrong
print("\n=== MANUAL TEST ===")
# Test with the first percentage
test_p = '01'
test_dir_path = os.path.join(base_dir, test_p, 'cBNs')
test_file_path = os.path.join(test_dir_path, "contingency_table_01.csv")

if os.path.exists(test_file_path):
    with open(test_file_path, 'r') as f:
        content = f.read()
        print(f"First 200 chars of {test_file_path}:")
        print(content[:200])
        print("\nLooking for tie patterns:")
        for j in range(6):
            pattern = re.compile(rf'Number of groups with exactly\s+{j+1}\s+best interventions:\s+(\d+)')
            match = pattern.search(content)
            if match:
                print(f"Found {j+1} ties: {match.group(1)}")
            else:
                print(f"NOT found {j+1} ties")
        
        print("\nLooking for action patterns:")
        action_pattern = re.compile(r'(\w+(?:_\w+)*)\s*:\s*Selected\s+(\d+)\s*times,\s*Safe\s+(\d+)\s*times,\s*Unsafe\s+(\d+)\s*times')
        matches = action_pattern.findall(content)
        for match in matches:
            print(f"Action match: {match}")
        
        print("\nLooking for transition patterns:")
        transitions = ['True -> False', 'True -> True', 'False -> False', 'False -> True']
        for trans in transitions:
            pattern = re.compile(rf'{trans}:\s+(\d+)')
            match = pattern.search(content)
            if match:
                print(f"Found {trans}: {match.group(1)}")
            else:
                print(f"NOT found {trans}")
else:
    print(f"Test file not found: {test_file_path}")
    print(f"Looking in: {os.path.abspath(test_dir_path)}")
    if os.path.exists(test_dir_path):
        print(f"Files in directory: {os.listdir(test_dir_path)}")

def generate_latex_files():
    try:
        # File 1: Detailed results by section
        with open('detailed_results.tex', 'w') as f:
            print("Generating detailed_results.tex...")
            f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\title{Tablas TPLP 2025}
\author{havilesa}
\date{July 2025}
\begin{document}
\maketitle

\section{Crashes}

\begin{table}[h]
\centering
\caption{Safe and Unsafe Outcomes by Data Training Percentage (Crash Scenarios)}
\begin{tabular}{l|ccccc|c}
\toprule
\multirow{2}{*}{Outcome} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
Safe (True $\rightarrow$ False) & """)
            f.write(" & ".join(str(crash_safety_data[p]['Safe']) for p in percentages))
            f.write(f" & {sum(crash_safety_data[p]['Safe'] for p in percentages)} \\\\ \n")
            f.write(r"""Unsafe (True $\rightarrow$ True) & """)
            f.write(" & ".join(str(crash_safety_data[p]['Unsafe']) for p in percentages))
            f.write(f" & {sum(crash_safety_data[p]['Unsafe'] for p in percentages)} \\\\ \n")
            f.write(r"""\midrule
Total & """)
            f.write(" & ".join(str(crash_safety_data[p]['Safe'] + crash_safety_data[p]['Unsafe']) for p in percentages))
            f.write(f""" & {sum(crash_safety_data[p]['Safe'] + crash_safety_data[p]['Unsafe'] for p in percentages)} \\\\ \n
\\bottomrule
\end{{tabular}}
\end{{table}}
""")
            print("Finished first table (Safe and Unsafe Outcomes, Crashes).")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Number of Ties by Data Training Percentage (Crash Scenarios)}
\begin{tabular}{c|ccccc|c}
\toprule
\multirow{2}{*}{\# of ties (1st Place)} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
""")
            for i in range(6):
                row_total = sum(crash_ties_data[p][i] for p in percentages)
                f.write(f"{i+1} & {crash_ties_data['01'][i]} & {crash_ties_data['25'][i]} & {crash_ties_data['50'][i]} & {crash_ties_data['75'][i]} & {crash_ties_data['90'][i]} & {row_total} \\\\ \n")
            column_totals = [sum(crash_ties_data[p]) for p in percentages]
            grand_total = sum(column_totals)
            f.write(f"""\midrule
Total & {' & '.join(map(str, column_totals))} & {grand_total} \\
\\bottomrule
\end{{tabular}}
\end{{table}}
""")
            print("Finished second table (Ties, Crashes).")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Number of Actions Selected by Data Training Percentage (Crash Scenarios)}
\begin{tabular}{l|ccccc|c|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} & \multirow{2}{*}{Safe} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & & \\
\midrule
""")
            for action in valid_actions:
                action_tex = action.replace('_', '\\_')
                row = f"\\texttt{{{action_tex}}}"
                for p in percentages:
                    row += f" & {crash_actions_data[p].get(action, 0)}"
                row += f" & {sum(crash_actions_data[p].get(action, 0) for p in percentages)} & {sum(crash_action_safety_data[p].get(action, 0) for p in percentages)} \\\\ \n"
                f.write(row)
            f.write(r"""\midrule
\textbf{Total}""")
            for p in percentages:
                f.write(f" & {sum(crash_actions_data[p].values())}")
            f.write(f" & {sum(sum(crash_actions_data[p].values()) for p in percentages)} & {sum(sum(crash_action_safety_data[p].values()) for p in percentages)} \\\\ \n")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}

\pagebreak
\section{No Crashes}

\begin{table}[h]
\centering
\caption{Safe and Unsafe Outcomes by Data Training Percentage (No Crash Scenarios)}
\begin{tabular}{l|ccccc|c}
\toprule
\multirow{2}{*}{Outcome} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
Safe (False $\rightarrow$ False) & """)
            f.write(" & ".join(str(no_crash_transition_counts[p]['False_False']) for p in percentages))
            f.write(f" & {sum(no_crash_transition_counts[p]['False_False'] for p in percentages)} \\\\ \n")
            f.write(r"""Unsafe (False $\rightarrow$ True) & """)
            f.write(" & ".join(str(no_crash_transition_counts[p]['False_True']) for p in percentages))
            f.write(f" & {sum(no_crash_transition_counts[p]['False_True'] for p in percentages)} \\\\ \n")
            f.write(r"""\midrule
Total & """)
            f.write(" & ".join(str(no_crash_transition_counts[p]['False_False'] + no_crash_transition_counts[p]['False_True']) for p in percentages))
            f.write(f""" & {sum(no_crash_transition_counts[p]['False_False'] + no_crash_transition_counts[p]['False_True'] for p in percentages)} \\\\ \n
\\bottomrule
\end{{tabular}}
\end{{table}}
""")
            print("Finished fourth table (Safe and Unsafe Outcomes, No Crashes).")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Number of Ties by Data Training Percentage (No Crash Scenarios)}
\begin{tabular}{c|ccccc|c}
\toprule
\multirow{2}{*}{\# of ties (1st Place)} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & \\
\midrule
""")
            for i in range(6):
                row_total = sum(no_crash_ties_data[p][i] for p in percentages)
                f.write(f"{i+1} & {no_crash_ties_data['01'][i]} & {no_crash_ties_data['25'][i]} & {no_crash_ties_data['50'][i]} & {no_crash_ties_data['75'][i]} & {no_crash_ties_data['90'][i]} & {row_total} \\\\ \n")
            column_totals = [sum(no_crash_ties_data[p]) for p in percentages]
            grand_total = sum(column_totals)
            f.write(f"""\midrule
Total & {' & '.join(map(str, column_totals))} & {grand_total} \\
\\bottomrule
\end{{tabular}}
\end{{table}}
""")
            print("Finished fifth table (Ties, No Crashes).")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Number of Actions Selected by Data Training Percentage (No Crash Scenarios)}
\begin{tabular}{l|ccccc|c|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{5}{c|}{Data training percentage} & \multirow{2}{*}{Total} & \multirow{2}{*}{Safe} \\
\cmidrule{2-6}
 & 01 & 25 & 50 & 75 & 90 & & \\
\midrule
""")
            for action in valid_actions:
                action_tex = action.replace('_', '\\_')
                row = f"\\texttt{{{action_tex}}}"
                for p in percentages:
                    row += f" & {no_crash_actions_data[p].get(action, 0)}"
                row += f" & {sum(no_crash_actions_data[p].get(action, 0) for p in percentages)} & {sum(no_crash_action_safety_data[p].get(action, 0) for p in percentages)} \\\\ \n"
                f.write(row)
            f.write(r"""\midrule
\textbf{Total}""")
            for p in percentages:
                f.write(f" & {sum(no_crash_actions_data[p].values())}")
            f.write(f" & {sum(sum(no_crash_actions_data[p].values()) for p in percentages)} & {sum(sum(no_crash_action_safety_data[p].values()) for p in percentages)} \\\\ \n")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}

\end{document}
""")
            print("Finished sixth table (Actions, No Crashes).")

        # File 2: Summary results with accumulated data
        with open('summary_results.tex', 'w') as f:
            print("Generating summary_results.tex...")
            f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\title{Tablas TPLP 2025}
\author{havilesa}
\date{July 2025}
\begin{document}
\maketitle

\begin{table}[h]
\centering
\caption{Accumulated Safe and Unsafe Outcomes Across All Percentages}
\begin{tabular}{l|cc}
\toprule
Category & Crash Scenarios & No Crash Scenarios \\
\midrule
Safe Interventions & """)
            f.write(f"{sum(crash_safety_data[p]['Safe'] for p in percentages)} & {sum(no_crash_transition_counts[p]['False_False'] for p in percentages)} \\\\ \n")
            f.write(r"""Unsafe Interventions & """)
            f.write(f"{sum(crash_safety_data[p]['Unsafe'] for p in percentages)} & {sum(no_crash_transition_counts[p]['False_True'] for p in percentages)} \\\\ \n")
            f.write(r"""\midrule
Total & """)
            f.write(f"{sum(crash_safety_data[p]['Safe'] + crash_safety_data[p]['Unsafe'] for p in percentages)} & {sum(no_crash_transition_counts[p]['False_False'] + no_crash_transition_counts[p]['False_True'] for p in percentages)} \\\\ \n")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}
""")
            print("Finished first table (Accumulated Safe and Unsafe Outcomes).")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Accumulated Number of Ties Across All Percentages}
\begin{tabular}{c|cc}
\toprule
\multirow{2}{*}{\# of ties (1st Place)} & \multicolumn{2}{c|}{Scenario} \\
\cmidrule{2-3}
 & Crash Scenarios & No Crash Scenarios \\
\midrule
""")
            for i in range(6):
                crash_total = sum(crash_ties_data[p][i] for p in percentages)
                no_crash_total = sum(no_crash_ties_data[p][i] for p in percentages)
                f.write(f"{i+1} & {crash_total} & {no_crash_total} \\\\ \n")
            crash_grand_total = sum(sum(crash_ties_data[p]) for p in percentages)
            no_crash_grand_total = sum(sum(no_crash_ties_data[p]) for p in percentages)
            f.write(f"""\midrule
Total & {crash_grand_total} & {no_crash_grand_total} \\
\\bottomrule
\end{{tabular}}
\end{{table}}
""")
            print("Finished second table (Accumulated Ties).")

            f.write(r"""
\begin{table}[h]
\centering
\caption{Accumulated Number of Actions Across All Percentages}
\begin{tabular}{l|cc|c}
\toprule
\multirow{2}{*}{Action} & \multicolumn{2}{c|}{Scenario} & \multirow{2}{*}{Total} \\
\cmidrule{2-3}
 & Crash Scenarios & No Crash Scenarios \\
\midrule
""")
            for action in valid_actions:
                action_tex = action.replace('_', '\\_')
                crash_total = sum(crash_actions_data[p].get(action, 0) for p in percentages)
                crash_safe = sum(crash_action_safety_data[p].get(action, 0) for p in percentages)
                no_crash_total = sum(no_crash_actions_data[p].get(action, 0) for p in percentages)
                no_crash_safe = sum(no_crash_action_safety_data[p].get(action, 0) for p in percentages)
                f.write(f"\\texttt{{{action_tex}}} & {crash_total} ({crash_safe} safe) & {no_crash_total} ({no_crash_safe} safe) & {crash_total + no_crash_total} \\\\ \n")
            f.write(r"""\midrule
\textbf{Total}""")
            crash_action_total = sum(sum(crash_actions_data[p].values()) for p in percentages)
            crash_safe_total = sum(sum(crash_action_safety_data[p].values()) for p in percentages)
            no_crash_action_total = sum(sum(no_crash_actions_data[p].values()) for p in percentages)
            no_crash_safe_total = sum(sum(no_crash_action_safety_data[p].values()) for p in percentages)
            f.write(f" & {crash_action_total} ({crash_safe_total} safe) & {no_crash_action_total} ({no_crash_safe_total} safe) & {crash_action_total + no_crash_action_total} \\\\ \n")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}

\section{Final Summary}

\begin{table}[h]
\centering
\caption{Final Summary of Random Selection Outcomes}
\begin{tabular}{lc}
\toprule
Outcome & Count \\
\midrule
Crash unsafe (True $\rightarrow$ True) & """)
            f.write(f"{final_summary['True_True']} \\\\ \n")
            f.write(r"""Crash safe (True $\rightarrow$ False) & """)
            f.write(f"{final_summary['True_False']} \\\\ \n")
            f.write(r"""No crash safe (False $\rightarrow$ False) & """)
            f.write(f"{final_summary['False_False']} \\\\ \n")
            f.write(r"""No crash unsafe (False $\rightarrow$ True) & """)
            f.write(f"{final_summary['False_True']} \\\\ \n")
            f.write(r"""\midrule
Total & """)
            f.write(f"{sum(final_summary.values())} \\\\ \n")
            f.write(r"""\bottomrule
\end{{tabular}}
\end{{table}}

\end{document}
""")
            print("Finished third table (Accumulated Actions).")
            print("Finished fourth table (Final Summary).")

        print("LaTeX files generated successfully:")
        print("- detailed_results.tex")
        print("- summary_results.tex")

    except Exception as e:
        print(f"Error generating LaTeX files: {str(e)}")
        raise

# Generate the LaTeX files
generate_latex_files()
