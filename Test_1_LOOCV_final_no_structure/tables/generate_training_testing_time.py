#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd

# Configuración
percentages = ['01', '25', '50', '75', '90']
base_dirs = ['../rep_1','../rep_2','../rep_3','../rep_4','../rep_5']

# Inicializar estructuras
training_times = {p: [] for p in percentages}
testing_times = {p: [] for p in percentages}
missing_files = []

# Función segura para convertir a float
def safe_float(s):
    try:
        return float(s.strip())
    except:
        return np.nan

# Procesar archivos
for base_dir in base_dirs:
    for p in percentages:
        # --- Training times ---
        training_file = os.path.join(base_dir, p, 'cBNs', 'training_numeralia.txt')
        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                content = f.read()
                # Extraer todos los tiempos de cada fold
                matches = re.findall(r'Training Time\s*=\s*([0-9.]+)\s*seconds', content)
                fold_times = [safe_float(m) for m in matches]
                training_times[p].extend(fold_times)
        else:
            missing_files.append(training_file)

        # --- Testing times from twin_networks_results.csv ---
        testing_file = os.path.join(base_dir, p, 'cBNs', 'twin_networks_results.csv')
        if os.path.exists(testing_file):
            try:
                df = pd.read_csv(testing_file)
                if 'elapsed_time' in df.columns:
                    testing_times[p].extend(df['elapsed_time'].astype(float).tolist())
                else:
                    print(f"Warning: 'elapsed_time' column not found in {testing_file}")
            except Exception as e:
                print(f"Error reading {testing_file}: {e}")
        else:
            missing_files.append(testing_file)

# Reporte de archivos faltantes
if missing_files:
    print("Archivos faltantes:")
    for f in missing_files:
        print(f" - {f}")
else:
    print("Todos los archivos de training y testing fueron encontrados.")

# Generar tabla LaTeX solo con los tiempos existentes
latex_file = "training_testing_summary.tex"
with open(latex_file, 'w') as f:
    f.write(r"\begin{table}[h]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Average Training and Testing Time per Data Training Percentage Across All Repetitions}" + "\n")
    f.write(r"\begin{tabular}{c|cc|cc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"Percentage & Avg. Training Time (s) & Std. Dev. (s) & Avg. Testing Time (s) & Std. Dev. (s) \\" + "\n")
    f.write(r"\midrule" + "\n")
    
    for p in percentages:
        train_avg = np.mean(training_times[p]) if training_times[p] else float('nan')
        train_std = np.std(training_times[p]) if training_times[p] else float('nan')
        test_avg = np.mean(testing_times[p]) if testing_times[p] else float('nan')
        test_std = np.std(testing_times[p]) if testing_times[p] else float('nan')
        f.write(f"{p} & {train_avg:.4f} & {train_std:.4f} & {test_avg:.4f} & {test_std:.4f} \\\\" + "\n")
    
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")

print(f"Archivo LaTeX generado: {latex_file}")

