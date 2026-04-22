library(data.table)
library(bnlearn)
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("Rgraphviz")

# ============================================================
# 1. LOAD DATA
# ============================================================

dt <- fread("./Shared_CSVs/integrated_DB_auto_humans_swerve.csv")
df <- as.data.frame(lapply(dt, factor))

target <- "latent_collision"

# ============================================================
# 2. RUN MMPC
# ============================================================

net_mmpc <- mmpc(
  df,
  test = "mi",        # match your CI tests
  alpha = 0.05,
  max.sx = 4
)

# ============================================================
# 3. BASIC OUTPUT
# ============================================================

cat("\n--- Learned Skeleton ---\n")
print(net_mmpc)

cat("\n--- Arcs ---\n")
print(arcs(net_mmpc))

cat("\n--- Number of edges ---\n")
print(nrow(arcs(net_mmpc)))

# ============================================================
# 4. LC CONNECTIVITY
# ============================================================

lc_neighbors <- nbr(net_mmpc, target)

cat("\n--- Neighbors of latent_collision ---\n")
print(lc_neighbors)
cat("Number of neighbors:", length(lc_neighbors), "\n")

# ============================================================
# 5. CHECK ACTION EDGE
# ============================================================

if ("action" %in% lc_neighbors) {
  cat("\n[OK] action is connected to latent_collision\n")
} else {
  cat("\n[WARNING] action is NOT connected to latent_collision\n")
}

# ============================================================
# 6. OPTIONAL: PLOT
# ============================================================

pdf("mmpc_graph.pdf", width=10, height=8)

graphviz.plot(net_mmpc)

dev.off()
