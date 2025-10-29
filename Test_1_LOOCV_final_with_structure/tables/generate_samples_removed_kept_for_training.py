#!/usr/bin/env python3
import os
import re
import numpy as np

# Configuración
percentages = ['01', '25', '50', '75', '90']
base_dirs = ['../rep_1','../rep_2','../rep_3','../rep_4','../rep_5']

# Inicializar estructuras
samples_removed = {p: [] for p in percentages}
train_sizes = {p: [] for p in percentages}
missing_files = []

# Función segura para convertir a int
def safe_int(s):
    try:
        return int(s.strip())
    except:
        return np.nan

# Procesar archivos
for base_dir in base_dirs:
    for p in percentages:
        # --- Training numeralia ---
        training_file = os.path.join(base_dir, p, 'cBNs', 'training_numeralia.txt')
        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                content = f.read()
                
                # Extraer Samples Removed
                removed_matches = re.findall(r'Samples Removed\s*=\s*([0-9]+)', content)
                samples_removed[p].extend([safe_int(m) for m in removed_matches])
                
                # Extraer Train Sample Size
                size_matches = re.findall(r'Train Sample Size\s*=\s*([0-9]+)', content)
                train_sizes[p].extend([safe_int(m) for m in size_matches])
        else:
            missing_files.append(training_file)

# Reporte de archivos faltantes
if missing_files:
    print("Archivos faltantes:")
    for f in missing_files:
        print(f" - {f}")
else:
    print("Todos los archivos de training fueron encontrados.")

# Generar tabla LaTeX con promedio y desviación estándar
latex_file = "training_summary.tex"
with open(latex_file, 'w') as f:
    f.write(r"\begin{table}[h]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Average Samples Removed and Train Sample Size per Percentage}" + "\n")
    f.write(r"\begin{tabular}{c|cc|cc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"Percentage & Avg. Removed & Std. Removed & Avg. Train Size & Std. Train Size \\" + "\n")
    f.write(r"\midrule" + "\n")
    
    for p in percentages:
        removed_avg = np.mean(samples_removed[p]) if samples_removed[p] else float('nan')
        removed_std = np.std(samples_removed[p]) if samples_removed[p] else float('nan')
        size_avg = np.mean(train_sizes[p]) if train_sizes[p] else float('nan')
        size_std = np.std(train_sizes[p]) if train_sizes[p] else float('nan')
        
        f.write(f"{p} & {removed_avg:.0f} & {removed_std:.0f} & {size_avg:.0f} & {size_std:.0f} \\\\" + "\n")
    
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")

print(f"Archivo LaTeX generado: {latex_file}")

