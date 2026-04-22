#!/usr/bin/env python3
# =============================================================================
# diagnose_lc_orientation.py
#
# Diagnoses the structural role of latent_collision in each learned .pl model
# across all sensitivity analysis combo directories.
#
# For each fold's cBN.pl file, classifies latent_collision into one of four
# categories based on its appearance in the probabilistic logic program:
#
#   Cat 1 — Correct (pure outcome):
#       LC appears as rule head AND never appears in body of other rules.
#       This is the desired structure for counterfactual reasoning.
#
#   Cat 2 — Intermediate node:
#       LC appears as rule head AND also appears in body of other rules.
#       LC is "in the middle" — connected but not a true outcome variable.
#       Counterfactual interventions on action may not reach LC.
#
#   Cat 3 — Pure parent (misoriented):
#       LC never appears as rule head BUT appears in body of other rules.
#       LC causes other variables — intervention on action cannot affect LC.
#
#   Cat 4 — Fully disconnected:
#       LC appears neither as head nor in body of any rule.
#       Equivalent to MMPC skeleton disconnection.
#
# Reports per-combo: category counts and percentages across 78 folds,
# plus which variables LC is a parent of (Cat 2/3) for interpretation.
#
# Usage:
#   python3 diagnose_lc_orientation.py [--base_dir .]
#
# Output:
#   lc_orientation_summary.csv   — machine-readable per-combo summary
#   lc_orientation_details.csv   — per-fold details
#   lc_orientation_report.txt    — human-readable report
# =============================================================================

import os
import re
import sys
import csv
import pandas as pd
from collections import defaultdict

# =============================================================================
# Combo directories to scan — same structure as summarize_sensitivity_results.py
# =============================================================================
MODEL_COMBOS = [
    # (model_label, subdir, config_id, combo_dir_name)

    # HC+BIC constrained
    ("HC+BIC constrained", "LOOCV_HC+BIC_constrained", "C1",
     "hc_bic_restart0_perturb1_maxpInf_fracp25"),
    ("HC+BIC constrained", "LOOCV_HC+BIC_constrained", "C2",
     "hc_bic_restart5_perturb1_maxpInf_fracp25"),
    ("HC+BIC constrained", "LOOCV_HC+BIC_constrained", "C3",
     "hc_bic_restart10_perturb1_maxpInf_fracp25"),
    ("HC+BIC constrained", "LOOCV_HC+BIC_constrained", "C4",
     "hc_bic_restart0_perturb1_maxp3_fracp25"),
    ("HC+BIC constrained", "LOOCV_HC+BIC_constrained", "C5",
     "hc_bic_restart0_perturb1_maxp5_fracp25"),
    ("HC+BIC constrained", "LOOCV_HC+BIC_constrained", "C6",
     "hc_bic_restart0_perturb2_maxpInf_fracp25"),
    ("HC+BIC constrained", "LOOCV_HC+BIC_constrained", "C7",
     "hc_aic_restart0_perturb1_maxpInf_fracp25"),

    # HC+BIC unconstrained
    ("HC+BIC unconstrained", "LOOCV_HC+BIC_unconstrained", "C1",
     "hc_bic_restart0_perturb1_maxpInf_fracp25"),
    ("HC+BIC unconstrained", "LOOCV_HC+BIC_unconstrained", "C2",
     "hc_bic_restart5_perturb1_maxpInf_fracp25"),
    ("HC+BIC unconstrained", "LOOCV_HC+BIC_unconstrained", "C3",
     "hc_bic_restart10_perturb1_maxpInf_fracp25"),
    ("HC+BIC unconstrained", "LOOCV_HC+BIC_unconstrained", "C4",
     "hc_bic_restart0_perturb1_maxp3_fracp25"),
    ("HC+BIC unconstrained", "LOOCV_HC+BIC_unconstrained", "C5",
     "hc_bic_restart0_perturb1_maxp5_fracp25"),
    ("HC+BIC unconstrained", "LOOCV_HC+BIC_unconstrained", "C6",
     "hc_bic_restart0_perturb2_maxpInf_fracp25"),
    ("HC+BIC unconstrained", "LOOCV_HC+BIC_unconstrained", "C7",
     "hc_aic_restart0_perturb1_maxpInf_fracp25"),

    # MMPC+HC+BIC
    ("MMPC+HC+BIC", "LOOCV_MMPC+HC+BIC", "C1",
     "mmpc_mi_alpha0p05_maxsx3_fracp25"),
    ("MMPC+HC+BIC", "LOOCV_MMPC+HC+BIC", "C2",
     "mmpc_mi_alpha0p01_maxsx3_fracp25"),
    ("MMPC+HC+BIC", "LOOCV_MMPC+HC+BIC", "C3",
     "mmpc_mi_alpha0p10_maxsx3_fracp25"),
    ("MMPC+HC+BIC", "LOOCV_MMPC+HC+BIC", "C4",
     "mmpc_mi_alpha0p05_maxsx2_fracp25"),
    ("MMPC+HC+BIC", "LOOCV_MMPC+HC+BIC", "C5",
     "mmpc_mi_alpha0p05_maxsx4_fracp25"),
    ("MMPC+HC+BIC", "LOOCV_MMPC+HC+BIC", "C6",
     "mmpc_mi_alpha0p01_maxsx2_fracp25"),
    ("MMPC+HC+BIC", "LOOCV_MMPC+HC+BIC", "C7",
     "mmpc_x2_alpha0p05_maxsx3_fracp25"),

    # PC unconstrained
    ("PC unconstrained", "LOOCV_PC_unconstrained", "C0",
     "pc_mi_alpha0p01_maxsxNULL_fracp25"),
    ("PC unconstrained", "LOOCV_PC_unconstrained", "C1",
     "pc_mi_alpha0p05_maxsx3_fracp25"),
    ("PC unconstrained", "LOOCV_PC_unconstrained", "C2",
     "pc_mi_alpha0p01_maxsx3_fracp25"),
    ("PC unconstrained", "LOOCV_PC_unconstrained", "C3",
     "pc_mi_alpha0p10_maxsx3_fracp25"),
    ("PC unconstrained", "LOOCV_PC_unconstrained", "C4",
     "pc_mi_alpha0p05_maxsx2_fracp25"),
    ("PC unconstrained", "LOOCV_PC_unconstrained", "C5",
     "pc_mi_alpha0p05_maxsx4_fracp25"),
    ("PC unconstrained", "LOOCV_PC_unconstrained", "C6",
     "pc_mi_alpha0p05_maxsxNULL_fracp25"),
    ("PC unconstrained", "LOOCV_PC_unconstrained", "C7",
     "pc_x2_alpha0p05_maxsx3_fracp25"),
]

# All variable names in the domain
ALL_VARS = [
    "action", "curr_lane", "free_E", "free_NE", "free_NW",
    "free_SE", "free_SW", "free_W", "latent_collision"
]


# =============================================================================
# Core parser: classify LC role in a single .pl file
# =============================================================================
def classify_lc(pl_path):
    """
    Parse a .pl file and classify the role of latent_collision.

    Returns a dict with:
      category     : 1, 2, 3, or 4 (see module docstring)
      as_child     : True if LC appears as rule head (i.e. LC has parents)
      as_parent    : True if LC appears in body of other variables' rules
      lc_parents   : list of variables that are parents of LC
      lc_children  : list of variables that LC is a parent of
      n_lc_rules   : number of rules where LC is the head
    """
    try:
        with open(pl_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return None

    lines = content.splitlines()

    # --- Detect LC as child (head of rules) ---
    # Matches: "latent_collision :- ..."  or  "P::uN.\nlatent_collision :- ..."
    lc_head_pattern = re.compile(r'^latent_collision\s*:-\s*(.+)', re.MULTILINE)
    lc_head_matches = lc_head_pattern.findall(content)
    as_child  = len(lc_head_matches) > 0
    n_lc_rules = len(lc_head_matches)

    # Extract parents of LC from rule bodies
    lc_parents = set()
    for body in lc_head_matches:
        for var in ALL_VARS:
            if var == "latent_collision":
                continue
            # Match var in body — check it's not preceded by \+
            for m in re.finditer(rf'\b{re.escape(var)}\b', body):
                # Look at the characters before the match to check for negation
                preceding = body[:m.start()].rstrip()
                if not preceding.endswith('\\+'):
                    lc_parents.add(var)
                    break

    # LC is only truly a child if it has at least one DOMAIN variable as parent.
    # If its only "parent" is a noise variable (uN), it is a root node —
    # structurally disconnected. HC+BIC makes LC a root after MMPC excludes
    # all edges to it, producing rules like "latent_collision :- u211."
    # which the head pattern matches but should be classified as Cat4.
    as_child = as_child and len(lc_parents) > 0

    # --- Detect LC as parent (appears in body of other variables' rules) ---
    # Look for rules where LC appears in the body but the head is NOT LC
    lc_children = set()
    as_parent   = False

    # Split into rule blocks — each head is either "X :- ..." or "P::uN."
    # We look for lines of the form "some_var :- ..., latent_collision, ..."
    # or "some_var :- ..., \+ latent_collision, ..."
    other_head_with_lc = re.compile(
        r'^(?!latent_collision)(\w+(?:\(\w+\))?)\s*:-\s*(.*\blatent_collision\b.*)',
        re.MULTILINE
    )
    for m in other_head_with_lc.finditer(content):
        head_var = m.group(1)
        # Strip function notation e.g. action(keep) -> action
        head_var_base = head_var.split('(')[0]
        if head_var_base in ALL_VARS and head_var_base != "latent_collision":
            lc_children.add(head_var_base)
            as_parent = True

    # --- Classify ---
    if as_child and not as_parent:
        category = 1   # Correct — pure outcome
    elif as_child and as_parent:
        category = 2   # Intermediate node
    elif not as_child and as_parent:
        category = 3   # Pure parent (misoriented)
    else:
        category = 4   # Fully disconnected

    return {
        'category':    category,
        'as_child':    as_child,
        'as_parent':   as_parent,
        'lc_parents':  sorted(lc_parents),
        'lc_children': sorted(lc_children),
        'n_lc_rules':  n_lc_rules,
    }


CATEGORY_LABELS = {
    1: "Cat1: pure outcome (correct)",
    2: "Cat2: intermediate node",
    3: "Cat3: pure parent (misoriented)",
    4: "Cat4: disconnected",
}


# =============================================================================
# Process a single combo directory
# =============================================================================
def process_combo(base_dir, subdir, combo_name, n_folds=78):
    combo_dir  = os.path.join(base_dir, subdir, combo_name)
    fold_rows  = []
    cat_counts = defaultdict(int)
    children_counter = defaultdict(int)  # which vars LC is parent of

    for fi in range(1, n_folds + 1):
        pl_path = os.path.join(combo_dir, f"fold_{fi}", "cBN.pl")
        if not os.path.exists(pl_path):
            fold_rows.append({
                'fold': fi, 'category': None,
                'as_child': None, 'as_parent': None,
                'lc_parents': '', 'lc_children': '', 'n_lc_rules': 0
            })
            cat_counts['missing'] += 1
            continue

        result = classify_lc(pl_path)
        if result is None:
            fold_rows.append({
                'fold': fi, 'category': None,
                'as_child': None, 'as_parent': None,
                'lc_parents': '', 'lc_children': '', 'n_lc_rules': 0
            })
            cat_counts['error'] += 1
            continue

        cat_counts[result['category']] += 1
        for v in result['lc_children']:
            children_counter[v] += 1

        fold_rows.append({
            'fold':        fi,
            'category':    result['category'],
            'as_child':    result['as_child'],
            'as_parent':   result['as_parent'],
            'lc_parents':  ', '.join(result['lc_parents']),
            'lc_children': ', '.join(result['lc_children']),
            'n_lc_rules':  result['n_lc_rules'],
        })

    n_found = sum(v for k, v in cat_counts.items() if isinstance(k, int))
    action_as_child_count = children_counter.get('action', 0)

    return {
        'fold_rows':             fold_rows,
        'cat_counts':            dict(cat_counts),
        'n_found':               n_found,
        'children_counter':      dict(children_counter),
        'action_as_child_count': action_as_child_count,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    base_dir = "."
    for idx, arg in enumerate(sys.argv[1:], 1):
        if arg == "--base_dir" and idx < len(sys.argv) - 1:
            base_dir = sys.argv[idx + 1]

    print(f"Base directory: {os.path.abspath(base_dir)}\n")

    summary_rows = []
    detail_rows  = []
    report_lines = []

    def rpt(s=""): report_lines.append(s)

    rpt("=" * 80)
    rpt("LATENT_COLLISION ORIENTATION DIAGNOSTIC REPORT")
    rpt("=" * 80)

    prev_model = None

    for model_label, subdir, config_id, combo_name in MODEL_COMBOS:
        combo_path = os.path.join(base_dir, subdir, combo_name)

        if not os.path.isdir(combo_path):
            print(f"[Missing dir] {combo_path}")
            summary_rows.append({
                'Model': model_label, 'Config': config_id,
                'Combo': combo_name,
                'N_folds': 0,
                'Cat1_correct': 'N/A', 'Cat1_pct': 'N/A',
                'Cat2_intermediate': 'N/A', 'Cat2_pct': 'N/A',
                'Cat3_misoriented': 'N/A', 'Cat3_pct': 'N/A',
                'Cat4_disconnected': 'N/A', 'Cat4_pct': 'N/A',
                'Missing_folds': 'N/A',
                'LC_children_when_parent': 'N/A',
            })
            continue

        result = process_combo(base_dir, subdir, combo_name)
        n      = result['n_found']
        cc     = result['cat_counts']

        def cnt(k): return cc.get(k, 0)
        def pct(k): return f"{100*cnt(k)/n:.1f}%" if n > 0 else "N/A"

        # Console progress
        print(f"[OK] {model_label:25s} {config_id:3s} | "
              f"Cat1={cnt(1):3d} Cat2={cnt(2):3d} "
              f"Cat3={cnt(3):3d} Cat4={cnt(4):3d} | "
              f"missing={cc.get('missing',0)} | {combo_name}")

        # action as child of LC — the key metric for counterfactual blocking
        act_child_n   = result['action_as_child_count']
        act_child_pct = f"{100*act_child_n/n:.1f}%" if n > 0 else "N/A"

        summary_rows.append({
            'Model':              model_label,
            'Config':             config_id,
            'Combo':              combo_name,
            'N_folds':            n,
            'Cat1_correct':       cnt(1),
            'Cat1_pct':           pct(1),
            'Cat2_intermediate':  cnt(2),
            'Cat2_pct':           pct(2),
            'Cat3_misoriented':   cnt(3),
            'Cat3_pct':           pct(3),
            'Cat4_disconnected':  cnt(4),
            'Cat4_pct':           pct(4),
            'Missing_folds':      cc.get('missing', 0),
            'Action_child_of_LC_n':   act_child_n,
            'Action_child_of_LC_pct': act_child_pct,
        })

        # Per-fold detail rows
        for fr in result['fold_rows']:
            detail_rows.append({
                'Model':       model_label,
                'Config':      config_id,
                'Combo':       combo_name,
                **fr
            })

        # Report section
        if model_label != prev_model:
            if prev_model is not None:
                rpt()
            rpt()
            rpt(f"{'─'*80}")
            rpt(f"  {model_label}")
            rpt(f"{'─'*80}")
            prev_model = model_label

        rpt()
        rpt(f"  {config_id}  {combo_name}")
        rpt(f"  Folds analysed       : {n}  (missing: {cc.get('missing',0)})")
        rpt(f"  Cat1 correct         : {cnt(1):3d}  ({pct(1)})")
        rpt(f"  Cat2 intermed.       : {cnt(2):3d}  ({pct(2)})")
        rpt(f"  Cat3 misoriented     : {cnt(3):3d}  ({pct(3)})")
        rpt(f"  Cat4 disconnect      : {cnt(4):3d}  ({pct(4)})")
        rpt(f"  action child of LC   : {act_child_n:3d}  ({act_child_pct})")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    detail_df  = pd.DataFrame(detail_rows)

    csv_sum = os.path.join(base_dir, "lc_orientation_summary.csv")
    csv_det = os.path.join(base_dir, "lc_orientation_details.csv")
    txt_rep = os.path.join(base_dir, "lc_orientation_report.txt")

    summary_df.to_csv(csv_sum, index=False)
    detail_df.to_csv(csv_det,  index=False)

    rpt()
    rpt("=" * 80)
    with open(txt_rep, 'w') as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nSummary CSV : {csv_sum}")
    print(f"Details CSV : {csv_det}")
    print(f"Report TXT  : {txt_rep}")

    # ------------------------------------------------------------------
    # Print condensed console table
    # ------------------------------------------------------------------
    W = 100
    print("\n" + "=" * W)
    print(f"{'Model':<25} {'C':>3}  {'Cat1':>12}  {'Cat2':>12}  "
          f"{'Cat3':>12}  {'Cat4':>12}  {'action→LC blocked':>18}")
    print("=" * W)
    prev = None
    for _, row in summary_df.iterrows():
        if row['Model'] != prev:
            if prev is not None:
                print("-" * W)
            prev = row['Model']
        c1 = f"{row['Cat1_correct']}({row['Cat1_pct']})"
        c2 = f"{row['Cat2_intermediate']}({row['Cat2_pct']})"
        c3 = f"{row['Cat3_misoriented']}({row['Cat3_pct']})"
        c4 = f"{row['Cat4_disconnected']}({row['Cat4_pct']})"
        ac = f"{row['Action_child_of_LC_n']}({row['Action_child_of_LC_pct']})"
        print(f"{row['Model']:<25} {row['Config']:>3}  "
              f"{c1:>12}  {c2:>12}  {c3:>12}  {c4:>12}  {ac:>18}")
    print("=" * W)

    # ------------------------------------------------------------------
    # Save LaTeX table
    # ------------------------------------------------------------------
    tex_out = os.path.join(base_dir, "lc_orientation_table.tex")
    with open(tex_out, 'w') as f:
        f.write(_build_latex(summary_df))
    print(f"LaTeX table : {tex_out}")


# =============================================================================
# LaTeX builder
# =============================================================================
def _build_latex(summary_df):
    lines = []
    lines.append(r"% Auto-generated by diagnose_lc_orientation.py")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Structural role of \texttt{latent\_collision} in learned")
    lines.append(r"PLTNs across CSL methods and configurations.")
    lines.append(r"Cat1: pure outcome (correct); Cat2: intermediate node;")
    lines.append(r"Cat3: pure parent (misoriented); Cat4: disconnected.")
    lines.append(r"Values are fold counts with percentage in parentheses.")
    lines.append(r"\texttt{action} blocked: folds where \texttt{action} is a child")
    lines.append(r"of \texttt{latent\_collision}, blocking counterfactual queries.}")
    lines.append(r"\label{tab:lc_orientation}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{clccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Config} & \textbf{Parameters}"
        r" & \textbf{Cat1 correct}"
        r" & \textbf{Cat2 intermediate}"
        r" & \textbf{Cat3 misoriented}"
        r" & \textbf{Cat4 disconnected}"
        r" & \textbf{\texttt{action} blocked} \\"
    )
    lines.append(r"\midrule")

    prev_model = None
    for _, row in summary_df.iterrows():
        if row['Model'] != prev_model:
            if prev_model is not None:
                lines.append(r"\midrule")
            # Centered model name spanning all columns
            model_tex = str(row['Model']).replace('+', r'+').replace('_', r'\_')
            lines.append(
                r"\multicolumn{7}{c}{\textbf{\texttt{" + model_tex + r"}}} \\"
            )
            lines.append(r"\midrule")
            prev_model = row['Model']

        def fmt(n, p):
            """Format as N (P%) for LaTeX."""
            if n == 'N/A':
                return r"\textemdash"
            return f"{n} ({p})"

        c1 = fmt(row['Cat1_correct'],      row['Cat1_pct'])
        c2 = fmt(row['Cat2_intermediate'], row['Cat2_pct'])
        c3 = fmt(row['Cat3_misoriented'],  row['Cat3_pct'])
        c4 = fmt(row['Cat4_disconnected'], row['Cat4_pct'])
        ac = fmt(row['Action_child_of_LC_n'], row['Action_child_of_LC_pct'])

        # Build parameter label — strip LaTeX math for configs that use it
        # (already in LaTeX format from MODEL_COMBOS)
        # Config IDs like C0, C1 ... C7
        config = str(row['Config'])

        # Get parameter string from combo name — derive readable label
        combo = str(row['Combo'])
        # Build a short human-readable param string from combo name
        param = _combo_to_param(combo)

        lines.append(
            f"{config} & {param} & {c1} & {c2} & {c3} & {c4} & {ac} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _combo_to_param(combo):
    """Convert combo directory name to a readable LaTeX parameter string."""
    # Examples:
    #   hc_bic_restart0_perturb1_maxpInf_fracp25  -> restart=0, perturb=1, maxp=$\infty$, BIC
    #   mmpc_mi_alpha0p05_maxsx3_fracp25           -> $\alpha$=0.05, max.sx=3, MI
    #   pc_mi_alpha0p05_maxsx3_fracp25             -> $\alpha$=0.05, max.sx=3, MI
    #   pc_mi_alpha0p01_maxsxNULL_fracp25          -> $\alpha$=0.01, max.sx=NULL, MI

    import re

    # HC variants
    m = re.match(
        r'hc_(bic|aic)_restart(\d+)_perturb(\d+)_maxp(Inf|\d+)_fracp\d+',
        combo)
    if m:
        score   = m.group(1).upper()
        restart = m.group(2)
        perturb = m.group(3)
        maxp    = r'$\infty$' if m.group(4) == 'Inf' else m.group(4)
        return f"restart={restart}, perturb={perturb}, maxp={maxp}, {score}"

    # MMPC / PC variants
    m = re.match(
        r'(?:mmpc|pc)_(mi|mi_sh|x2|sp_mi)_alpha(\d+p\d+)_maxsx(NULL|\d+)_fracp\d+',
        combo)
    if m:
        test  = m.group(1).replace('_', '-').upper()
        alpha = m.group(2).replace('p', '.')
        maxsx = m.group(3)
        return rf'$\alpha$={alpha}, max.sx={maxsx}, {test}'

    return combo  # fallback: return raw name


if __name__ == "__main__":
    main()
