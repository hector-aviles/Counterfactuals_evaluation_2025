library(data.table)
library(bnlearn)

# ============================================================
# 1. LOAD DATA
# ============================================================

dt <- fread("./Shared_CSVs/integrated_DB_auto_humans_swerve.csv")
df <- as.data.frame(lapply(dt, factor))

target <- "latent_collision"
vars <- setdiff(names(df), target)

# ============================================================
# 2. MARGINAL TESTS
# ============================================================

results <- list()

for (v in vars) {
  test <- ci.test(target, v, data = df, test = "mi")
  
  results[[length(results)+1]] <- data.table(
    variable = v,
    cond_size = 0,
    conditioned_on = "",
    p_value = test$p.value
  )
}

# ============================================================
# 3. FUNCTION TO GENERATE CI TESTS
# ============================================================

run_ci_tests <- function(cond_size) {
  
  res <- list()
  
  for (v in vars) {
    
    cond_vars <- setdiff(vars, v)
    
    # Generate combinations of size k
    combs <- combn(cond_vars, cond_size, simplify = FALSE)
    
    for (Z in combs) {
      
      test <- ci.test(target, v, z = Z, data = df, test = "mi")
      
      res[[length(res)+1]] <- data.table(
        variable = v,
        cond_size = cond_size,
        conditioned_on = paste(Z, collapse = ","),
        p_value = test$p.value
      )
    }
  }
  
  return(rbindlist(res))
}

# ============================================================
# 4. RUN FOR SIZES 1–4
# ============================================================

for (k in 1:4) {
  cat("Running conditioning size =", k, "\n")
  results[[length(results)+1]] <- run_ci_tests(k)
}

results_dt <- rbindlist(results)

# ============================================================
# 5. SUMMARY PER VARIABLE AND CONDITION SIZE
# ============================================================

summary_dt <- results_dt[, .(
  min_p_value = min(p_value),
  mean_p_value = mean(p_value)
), by = .(variable, cond_size)]

setorder(summary_dt, variable, cond_size)

print(summary_dt)

# ============================================================
# 6. OPTIONAL: SAVE
# ============================================================

fwrite(summary_dt, "ci_test_summary_extended.csv")



