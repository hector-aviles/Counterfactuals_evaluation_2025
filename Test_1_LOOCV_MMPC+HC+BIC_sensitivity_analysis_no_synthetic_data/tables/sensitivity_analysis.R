library(data.table)

# ============================================================
# 1. LOAD DATA
# ============================================================

# Option A: load from CSV
dt <- fread("../sensitivity_results/all_combinations_summary.csv")

# Option B (if already in memory):
# dt <- copy(global_summary)

# ============================================================
# 2. COMPUTE NUMBER OF LC NEIGHBORS
# ============================================================

dt[, LC_n := ifelse(LC_neighbors == "NONE", 0,
                    lengths(strsplit(LC_neighbors, ",")))]

# ============================================================
# 3. AGGREGATE (MEAN + STD)
# ============================================================

summary_dt <- dt[, .(
  avg_LC = mean(LC_n),
  sd_LC  = sd(LC_n),

  avg_ACT = 100 * mean(Action_in_LC_neighbors),
  sd_ACT  = 100 * sd(Action_in_LC_neighbors),

  avg_EDG = mean(Skeleton_edges),
  sd_EDG  = sd(Skeleton_edges)

), by = .(MMPC_alpha, MMPC_max_sx, MMPC_test)]

# ============================================================
# 4. SORT BY MOST CONNECTED FIRST
# ============================================================

setorder(summary_dt, -avg_LC)

# ============================================================
# 5. FORMAT AS "MEAN ± STD"
# ============================================================

summary_dt[, LC  := sprintf("%.2f $\\pm$ %.2f", avg_LC, sd_LC)]
summary_dt[, ACT := sprintf("%.1f $\\pm$ %.1f", avg_ACT, sd_ACT)]
summary_dt[, EDG := sprintf("%.1f $\\pm$ %.1f", avg_EDG, sd_EDG)]

# Keep only formatted columns
summary_dt <- summary_dt[, .(
  MMPC_alpha,
  MMPC_max_sx,
  MMPC_test,
  LC,
  ACT,
  EDG
)]

# Rename for LaTeX (short names)
setnames(summary_dt,
         c("MMPC_alpha","MMPC_max_sx","MMPC_test","LC","ACT","EDG"),
         c("$\\alpha$","sx","test","LC","Act(\\%)","Edges"))

# ============================================================
# 6. GENERATE LATEX TABLE (ROTATED)
# ============================================================

cat("\\begin{table}[!tb]\n")
cat("\\centering\n")
cat("\\caption{MMPC sensitivity analysis. Values are mean $\\pm$ standard deviation across folds: size of the Markov Blanket of \\texttt{latent\\_collision}, percentage of folds where \\texttt{action} is connected to \\texttt{latent\\_collision}, and number of skeleton edges.}\n")

cat("\\begin{tabular}{cccccc}\n")
cat("\\toprule\n")

cat("& & & \\multicolumn{3}{c}{\\texttt{latent\\_collision} connectivity} \\\\\n")
cat("\\cmidrule(lr){4-6}\n")
cat("\\multicolumn{3}{c}{MMPC Parameters} & & & \\\\\n")
cat("\\cmidrule(lr){1-3}\n")

cat("$\\alpha$ & sx & test & MB size & \\smashedcell{\\texttt{action}-\\texttt{latent\\_collision} \\\\ edges (\\%)} & \\smashedcell{\\# of skeleton \\\\ edges} \\\\\n")

cat("\\midrule\n")

# Rows
for (i in 1:nrow(summary_dt)) {
  cat(paste(summary_dt[i], collapse = " & "), "\\\\\n")
}


cat("\\bottomrule\n")
cat("\\end{tabular}\n")
cat("\\label{tab:mmpc_sensitivity}\n")
cat("\\end{table}\n")
