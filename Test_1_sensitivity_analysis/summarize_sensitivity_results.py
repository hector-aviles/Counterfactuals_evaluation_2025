#!/usr/bin/env python3
# =============================================================================
# summarize_sensitivity_results.py
#
# Builds a summary table across all 4 model types and their 7 configurations.
# Reads data_sorted.csv from each combo directory and computes per-fold metrics,
# then reports mean ± std across the 78 folds per configuration.
#
# Metrics per fold:
#   - Safe actions    : ranking==1 rows where potential_crash_after_intervention=='False'
#   - Unsafe actions  : ranking==1 rows where potential_crash_after_intervention=='True'
#   - #Actions suggested : total ranking==1 rows (safe + unsafe, i.e. all tied best)
#   - Diversity       : number of distinct iactions at ranking==1
#
# Output:
#   sensitivity_summary.csv   — machine-readable
#   sensitivity_summary.tex   — LaTeX table
#
# Usage:
#   python3 summarize_sensitivity_results.py [--base_dir .]
#
# Expected directory structure:
#   <base_dir>/LOOCV_HC+BIC_constrained/<combo>/data_sorted.csv
#   <base_dir>/LOOCV_HC+BIC_unconstrained/<combo>/data_sorted.csv
#   <base_dir>/LOOCV_MMPC+HC+BIC/<combo>/data_sorted.csv
#   <base_dir>/LOOCV_PC_unconstrained/<combo>/data_sorted.csv
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np

# =============================================================================
# Model / config definitions
# Each entry: (model_label, subdir, config_id, param_label, combo_dir_name)
# =============================================================================
CONFIGS = [
    # ── HC+BIC constrained ────────────────────────────────────────────────
    ("HC+BIC\nconstrained", "LOOCV_HC+BIC_constrained",
     "C1", r"restart=0, perturb=1, maxp=$\infty$, BIC",
     "hc_bic_restart0_perturb1_maxpInf_fracp25"),
    ("HC+BIC\nconstrained", "LOOCV_HC+BIC_constrained",
     "C2", r"restart=5, perturb=1, maxp=$\infty$, BIC",
     "hc_bic_restart5_perturb1_maxpInf_fracp25"),
    ("HC+BIC\nconstrained", "LOOCV_HC+BIC_constrained",
     "C3", r"restart=10, perturb=1, maxp=$\infty$, BIC",
     "hc_bic_restart10_perturb1_maxpInf_fracp25"),
    ("HC+BIC\nconstrained", "LOOCV_HC+BIC_constrained",
     "C4", r"restart=0, perturb=1, maxp=3, BIC",
     "hc_bic_restart0_perturb1_maxp3_fracp25"),
    ("HC+BIC\nconstrained", "LOOCV_HC+BIC_constrained",
     "C5", r"restart=0, perturb=1, maxp=5, BIC",
     "hc_bic_restart0_perturb1_maxp5_fracp25"),
    ("HC+BIC\nconstrained", "LOOCV_HC+BIC_constrained",
     "C6", r"restart=0, perturb=2, maxp=$\infty$, BIC",
     "hc_bic_restart0_perturb2_maxpInf_fracp25"),
    ("HC+BIC\nconstrained", "LOOCV_HC+BIC_constrained",
     "C7", r"restart=0, perturb=1, maxp=$\infty$, AIC",
     "hc_aic_restart0_perturb1_maxpInf_fracp25"),

    # ── HC+BIC unconstrained ──────────────────────────────────────────────
    ("HC+BIC\nunconstrained", "LOOCV_HC+BIC_unconstrained",
     "C1", r"restart=0, perturb=1, maxp=$\infty$, BIC",
     "hc_bic_restart0_perturb1_maxpInf_fracp25"),
    ("HC+BIC\nunconstrained", "LOOCV_HC+BIC_unconstrained",
     "C2", r"restart=5, perturb=1, maxp=$\infty$, BIC",
     "hc_bic_restart5_perturb1_maxpInf_fracp25"),
    ("HC+BIC\nunconstrained", "LOOCV_HC+BIC_unconstrained",
     "C3", r"restart=10, perturb=1, maxp=$\infty$, BIC",
     "hc_bic_restart10_perturb1_maxpInf_fracp25"),
    ("HC+BIC\nunconstrained", "LOOCV_HC+BIC_unconstrained",
     "C4", r"restart=0, perturb=1, maxp=3, BIC",
     "hc_bic_restart0_perturb1_maxp3_fracp25"),
    ("HC+BIC\nunconstrained", "LOOCV_HC+BIC_unconstrained",
     "C5", r"restart=0, perturb=1, maxp=5, BIC",
     "hc_bic_restart0_perturb1_maxp5_fracp25"),
    ("HC+BIC\nunconstrained", "LOOCV_HC+BIC_unconstrained",
     "C6", r"restart=0, perturb=2, maxp=$\infty$, BIC",
     "hc_bic_restart0_perturb2_maxpInf_fracp25"),
    ("HC+BIC\nunconstrained", "LOOCV_HC+BIC_unconstrained",
     "C7", r"restart=0, perturb=1, maxp=$\infty$, AIC",
     "hc_aic_restart0_perturb1_maxpInf_fracp25"),

    # ── MMPC+HC+BIC ───────────────────────────────────────────────────────
    ("MMPC+\nHC+BIC", "LOOCV_MMPC+HC+BIC",
     "C1", r"$\alpha$=0.05, max.sx=3, MI",
     "mmpc_mi_alpha0p05_maxsx3_fracp25"),
    ("MMPC+\nHC+BIC", "LOOCV_MMPC+HC+BIC",
     "C2", r"$\alpha$=0.01, max.sx=3, MI",
     "mmpc_mi_alpha0p01_maxsx3_fracp25"),
    ("MMPC+\nHC+BIC", "LOOCV_MMPC+HC+BIC",
     "C3", r"$\alpha$=0.10, max.sx=3, MI",
     "mmpc_mi_alpha0p10_maxsx3_fracp25"),
    ("MMPC+\nHC+BIC", "LOOCV_MMPC+HC+BIC",
     "C4", r"$\alpha$=0.05, max.sx=2, MI",
     "mmpc_mi_alpha0p05_maxsx2_fracp25"),
    ("MMPC+\nHC+BIC", "LOOCV_MMPC+HC+BIC",
     "C5", r"$\alpha$=0.05, max.sx=4, MI",
     "mmpc_mi_alpha0p05_maxsx4_fracp25"),
    ("MMPC+\nHC+BIC", "LOOCV_MMPC+HC+BIC",
     "C6", r"$\alpha$=0.01, max.sx=2, MI",
     "mmpc_mi_alpha0p01_maxsx2_fracp25"),
    ("MMPC+\nHC+BIC", "LOOCV_MMPC+HC+BIC",
     "C7", r"$\alpha$=0.05, max.sx=3, $\chi^2$",
     "mmpc_x2_alpha0p05_maxsx3_fracp25"),

    # ── PC unconstrained ──────────────────────────────────────────────────
    ("PC\nunconstrained", "LOOCV_PC_unconstrained",
     "C0", r"$\alpha$=0.01, max.sx=NULL, MI (original)",
     "pc_mi_alpha0p01_maxsxNULL_fracp25"),
    ("PC\nunconstrained", "LOOCV_PC_unconstrained",
     "C1", r"$\alpha$=0.05, max.sx=3, MI",
     "pc_mi_alpha0p05_maxsx3_fracp25"),
    ("PC\nunconstrained", "LOOCV_PC_unconstrained",
     "C2", r"$\alpha$=0.01, max.sx=3, MI",
     "pc_mi_alpha0p01_maxsx3_fracp25"),
    ("PC\nunconstrained", "LOOCV_PC_unconstrained",
     "C3", r"$\alpha$=0.10, max.sx=3, MI",
     "pc_mi_alpha0p10_maxsx3_fracp25"),
    ("PC\nunconstrained", "LOOCV_PC_unconstrained",
     "C4", r"$\alpha$=0.05, max.sx=2, MI",
     "pc_mi_alpha0p05_maxsx2_fracp25"),
    ("PC\nunconstrained", "LOOCV_PC_unconstrained",
     "C5", r"$\alpha$=0.05, max.sx=4, MI",
     "pc_mi_alpha0p05_maxsx4_fracp25"),
    ("PC\nunconstrained", "LOOCV_PC_unconstrained",
     "C6", r"$\alpha$=0.05, max.sx=NULL, MI",
     "pc_mi_alpha0p05_maxsxNULL_fracp25"),
    ("PC\nunconstrained", "LOOCV_PC_unconstrained",
     "C7", r"$\alpha$=0.05, max.sx=3, $\chi^2$",
     "pc_x2_alpha0p05_maxsx3_fracp25"),
]


# =============================================================================
# Metrics computation from data_sorted.csv
# =============================================================================
def compute_metrics(ds_path):
    """
    Read data_sorted.csv and compute per-fold metrics.
    Returns a DataFrame with one row per fold (group_id).
    """
    try:
        df = pd.read_csv(ds_path, dtype=str)
    except Exception as e:
        print(f"  [Error] Cannot read {ds_path}: {e}")
        return None

    # Normalize numeric columns
    df['probability'] = pd.to_numeric(df['probability'], errors='coerce')
    df['group_id']    = pd.to_numeric(df['group_id'],    errors='coerce').astype(int)
    df['ranking']     = pd.to_numeric(df['ranking'],     errors='coerce').astype(int)

    # Normalize crash columns — may be bool or string
    for col in ['potential_crash_after_intervention']:
        if col in df.columns:
            df[col] = df[col].map({
                True: 'True', False: 'False',
                'True': 'True', 'False': 'False',
                'true': 'True', 'false': 'False'
            })

    if 'potential_crash_after_intervention' not in df.columns:
        print(f"  [Error] Missing potential_crash_after_intervention in {ds_path}")
        return None

    # Work only with ranking==1 rows (best action candidates per fold)
    r1 = df[df['ranking'] == 1].copy()

    fold_metrics = []
    for gid, grp in r1.groupby('group_id'):
        safe    = (grp['potential_crash_after_intervention'] == 'False').sum()
        unsafe  = (grp['potential_crash_after_intervention'] == 'True').sum()
        n_total = len(grp)
        diversity = grp['iaction'].nunique()
        fold_metrics.append({
            'group_id':  gid,
            'safe':      safe,
            'unsafe':    unsafe,
            'n_actions': n_total,
            'diversity': diversity
        })

    return pd.DataFrame(fold_metrics) if fold_metrics else None


def mean_std(series):
    """Return 'mean ± std' string formatted to 2 decimal places."""
    if len(series) == 0:
        return "N/A"
    m = series.mean()
    s = series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{m:.2f} ± {s:.2f}"


# =============================================================================
# Main
# =============================================================================
def main():
    # Parse --base_dir argument
    base_dir = "."
    for idx, arg in enumerate(sys.argv[1:], 1):
        if arg == "--base_dir" and idx < len(sys.argv) - 1:
            base_dir = sys.argv[idx + 1]

    print(f"Base directory: {os.path.abspath(base_dir)}\n")

    rows = []

    for model_label, subdir, config_id, param_label, combo_name in CONFIGS:
        ds_path = os.path.join(base_dir, subdir, combo_name, "data_sorted.csv")

        # Clean model label for display (remove newline)
        model_clean = model_label.replace("\n", " ")

        if not os.path.exists(ds_path):
            print(f"[Missing] {ds_path}")
            rows.append({
                'Model':      model_clean,
                'Config':     config_id,
                'Params':     param_label,
                'Safe':       "N/A",
                'Unsafe':     "N/A",
                'N_actions':  "N/A",
                'Diversity':  "N/A",
                'N_folds':    0,
                'combo':      combo_name,
            })
            continue

        metrics = compute_metrics(ds_path)
        if metrics is None or len(metrics) == 0:
            print(f"[Empty]   {ds_path}")
            rows.append({
                'Model':     model_clean,
                'Config':    config_id,
                'Params':    param_label,
                'Safe':      "N/A",
                'Unsafe':    "N/A",
                'N_actions': "N/A",
                'Diversity': "N/A",
                'N_folds':   0,
                'combo':     combo_name,
            })
            continue

        n_folds = len(metrics)
        print(f"[OK] {model_clean:25s} {config_id} | {n_folds} folds | {combo_name}")

        rows.append({
            'Model':     model_clean,
            'Config':    config_id,
            'Params':    param_label,
            'Safe':      mean_std(metrics['safe']),
            'Unsafe':    mean_std(metrics['unsafe']),
            'N_actions': mean_std(metrics['n_actions']),
            'Diversity': mean_std(metrics['diversity']),
            'N_folds':   n_folds,
            'combo':     combo_name,
        })

    summary = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_out = os.path.join(base_dir, "sensitivity_summary.csv")
    summary.to_csv(csv_out, index=False)
    print(f"\nCSV saved to: {csv_out}")

    # ------------------------------------------------------------------
    # Save LaTeX table
    # ------------------------------------------------------------------
    tex_out = os.path.join(base_dir, "sensitivity_summary.tex")
    with open(tex_out, "w") as f:
        f.write(_build_latex(summary))
    print(f"LaTeX saved to: {tex_out}")

    # ------------------------------------------------------------------
    # Print to console
    # ------------------------------------------------------------------
    W = 95
    print("\n" + "="*W)
    print(f"{'Config':<8}  {'α / params':<38}  {'Safe actions':>15}  {'Unsafe actions':>15}  {'Total actions':>15}  {'N':>4}")
    print("="*W)
    prev_model = None
    for _, row in summary.iterrows():
        if row['Model'] != prev_model:
            if prev_model is not None:
                print("="*W)
            # Centered model name separator
            label = f"  {row['Model']}  "
            pad   = (W - len(label)) // 2
            print(" "*pad + label)
            print("-"*W)
            prev_model = row['Model']
        # Truncate param label for console display
        params_short = row['Params'].replace('$\\infty$','Inf').replace('$\\chi^2$','x2').replace('$\\alpha$','a').replace('\\','')
        print(f"{row['Config']:<8}  {params_short:<38}  "
              f"{row['Safe']:>15}  {row['Unsafe']:>15}  "
              f"{row['N_actions']:>15}  "
              f"{row['N_folds']:>4}")
    print("="*W)


# =============================================================================
# LaTeX builder
# =============================================================================
def _build_latex(summary):
    lines = []
    lines.append(r"% Auto-generated by summarize_sensitivity_results.py")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sensitivity analysis summary across all CSL methods and parameter")
    lines.append(r"configurations. Values are mean $\pm$ std across 78 folds per configuration.")
    lines.append(r"Safe/Unsafe actions: number of ranking-1 actions labelled safe/unsafe per fold.")
    lines.append(r"Total actions: total ranking-1 actions including ties.}")
    lines.append(r"\label{tab:sensitivity_summary}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{clcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Config} & \textbf{$\alpha$ / params}"
                 r" & \textbf{Safe actions} & \textbf{Unsafe actions}"
                 r" & \textbf{Total actions} & \textbf{N} \\")
    lines.append(r"\midrule")

    prev_model = None
    for _, row in summary.iterrows():
        if row['Model'] != prev_model:
            # Insert separator between model groups
            if prev_model is not None:
                lines.append(r"\midrule")
            # Centered model name row spanning all columns
            model_tex = row['Model'].replace('+', r'+')
            lines.append(
                r"\multicolumn{6}{c}{\textbf{\texttt{" + model_tex + r"}}} \\"
            )
            lines.append(r"\midrule")
            prev_model = row['Model']

        param_tex = row['Params'].replace('_', r'\_').replace('%', r'\%')
        lines.append(
            f"{row['Config']} & {param_tex} & "
            f"{row['Safe']} & {row['Unsafe']} & "
            f"{row['N_actions']} & {row['N_folds']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
