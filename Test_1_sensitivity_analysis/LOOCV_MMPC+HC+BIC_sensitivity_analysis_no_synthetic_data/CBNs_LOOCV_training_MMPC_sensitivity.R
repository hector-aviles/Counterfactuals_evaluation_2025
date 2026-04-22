#!/usr/bin/env Rscript
# =============================================================================
# CBNs_LOOCV_training_MMPC_sensitivity.R
#
# Sensitivity analysis over MMPC hyperparameters.
# All combinations MUST receive the same --fold_ids to be comparable.
# The shell script run_sensitivity_analysis.sh handles this automatically.
#
# Parameters:
#   --alpha    : CI test significance threshold (default 0.05)
#   --max.sx   : maximum conditioning set size  (default 3)
#   --test     : CI test: "mi" (default) or "mi-sh"
#   --fold_ids : comma-separated fold indices (REQUIRED for comparability)
#   --folds N  : fallback — pick N stratified folds (only if --fold_ids absent)
#
# Output:
#   sensitivity_results/<COMBO_LABEL>/rep_<N>/<pct>/cBNs/
#   sensitivity_results/<COMBO_LABEL>/global_summary.csv
#
# COMBO_LABEL env var is set by the shell runner.
# =============================================================================

suppressMessages({
  for (pkg in c("data.table","conflicted","bnlearn",
                "dplyr","stringr","tidyr","tidyverse","arules")) {
    if (!requireNamespace(pkg, quietly=TRUE))
      install.packages(pkg, repos="https://cloud.r-project.org")
  }
})

library(data.table); library(conflicted)
conflict_prefer("setdiff","base")
library(bnlearn); library(dplyr); library(stringr)
library(tidyr);   library(tidyverse); library(arules)

dec_round <- 7L

# =============================================================================
# Argument parsing
# =============================================================================
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 1)
  stop("Usage: Rscript script.R <input_dir> [reps] [pcts] [--alpha V] [--max.sx V] [--test T] [--fold_ids id1,id2,...] [--folds N]")

input_subdir1   <- args[1]
reps_str        <- "1"
percentages_str <- "01"
mmpc_alpha      <- 0.05
mmpc_max_sx     <- 3L
mmpc_test       <- "mi"
fold_ids_arg    <- NULL   # explicit fold IDs (preferred)
n_folds_subset  <- NULL   # fallback: pick N stratified folds

pos <- 2L
if (length(args)>=pos && !startsWith(args[pos],"--")){ reps_str        <- args[pos]; pos <- pos+1L }
if (length(args)>=pos && !startsWith(args[pos],"--")){ percentages_str <- args[pos]; pos <- pos+1L }

named <- if (pos<=length(args)) args[pos:length(args)] else character(0)
i <- 1L
while (i <= length(named)) {
  fl <- named[i]
  if      (fl=="--alpha"    && i<length(named)){ mmpc_alpha    <- as.numeric(named[i+1]);  i<-i+2L }
  else if (fl=="--max.sx"   && i<length(named)){ mmpc_max_sx   <- as.integer(named[i+1]);  i<-i+2L }
  else if (fl=="--test"     && i<length(named)){ mmpc_test     <- named[i+1];              i<-i+2L }
  else if (fl=="--fold_ids" && i<length(named)){ fold_ids_arg  <- as.integer(strsplit(named[i+1],",")[[1]]); i<-i+2L }
  else if (fl=="--folds"    && i<length(named)){ n_folds_subset<- as.integer(named[i+1]); i<-i+2L }
  else { i<-i+1L }
}

if (!mmpc_test %in% c("mi","mi-sh","x2","sp-mi","smc-mi"))
  stop("Invalid --test '", mmpc_test, "'. Valid: mi, mi-sh, x2, sp-mi, smc-mi")

parse_range <- function(s) {
  if (grepl("-",s,fixed=TRUE)){ b<-as.integer(strsplit(s,"-")[[1]]); seq(b[1],b[2]) }
  else if (grepl(",",s,fixed=TRUE)){ as.integer(strsplit(s,",")[[1]]) }
  else { as.integer(s) }
}
reps        <- parse_range(reps_str)
percentages <- sprintf("%02d", as.integer(strsplit(percentages_str,",")[[1]] |> trimws()))

# Output root
combo_label  <- Sys.getenv("COMBO_LABEL", unset="")
if (nchar(combo_label)==0)
  combo_label <- sprintf("%s_alpha%.2f_maxsx%d", gsub("-","_",mmpc_test), mmpc_alpha, mmpc_max_sx)

base_wd      <- getwd()
results_root <- file.path(base_wd, "sensitivity_results", combo_label)
if (!dir.exists(results_root)) dir.create(results_root, recursive=TRUE)

message("=== Configuration ===")
message("Input dir    : ", input_subdir1)
message("Reps         : ", paste(reps, collapse=", "))
message("Percentages  : ", paste(percentages, collapse=", "))
message("CI test      : ", mmpc_test)
message("Alpha        : ", mmpc_alpha)
message("Max.sx       : ", mmpc_max_sx)
message("Combo label  : ", combo_label)
message("Results root : ", results_root)

# =============================================================================
# Seeds and shared paths
# =============================================================================
seeds           <- c(300L,456L,211L,26L,500L,1001L,724L,881L,91L,255L)
shared_csv_path <- "./Shared_CSVs"

adjust_values <- function(freq) {
  freq[is.na(freq)] <- 0.0; freq[freq<=1e-4] <- 1e-3
  s <- sum(freq)
  if (s==0) rep(1/length(freq),length(freq)) else if (s>1) freq/s else freq
}

# =============================================================================
# Load datasets
# =============================================================================
for (f in file.path(shared_csv_path,c("integrated_DB_auto_humans_swerve.csv","crashes.csv","no_crashes.csv")))
  if (!file.exists(f)) stop("Missing: ", f)

dt            <- fread(file.path(input_subdir1,"integrated_DB_auto_humans_swerve.csv"), colClasses="character")
dt            <- dt[complete.cases(dt)]
dt_crashes    <- fread(file.path(input_subdir1,"crashes.csv"),    colClasses="character")
dt_no_crashes <- fread(file.path(input_subdir1,"no_crashes.csv"), colClasses="character")
dt_crashes[,    orig_label_lc:="True"]
dt_no_crashes[, orig_label_lc:="False"]
dt_unique <- rbindlist(list(dt_crashes,dt_no_crashes))
dt_unique[, latent_collision:="True"]

state_cols <- c("curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W")
dt[,        state_id:=do.call(paste,c(.SD,sep="_")), .SDcols=state_cols]
dt_unique[, state_id:=do.call(paste,c(.SD,sep="_")), .SDcols=state_cols]

# =============================================================================
# Determine fold selection
# NOTE: --fold_ids is strongly preferred; all combinations MUST share the same IDs.
# =============================================================================
all_folds <- seq_len(nrow(dt_unique))

if (!is.null(fold_ids_arg)) {
  # Explicit IDs supplied by shell script — guaranteed identical across combos
  selected_folds <- sort(fold_ids_arg[fold_ids_arg %in% all_folds])
  message("Fold source  : --fold_ids (", length(selected_folds), " folds) — COMPARABLE")

} else if (!is.null(n_folds_subset)) {
  # Fallback: stratified random selection.
  # WARNING: without a shared seed this will differ across combos.
  warning("--fold_ids not supplied. Using --folds fallback. ",
          "Results may NOT be directly comparable across combinations!")
  set.seed(42L)   # fixed seed so at least repeated runs of the same combo are stable
  act_col <- dt_unique$action; acts <- unique(act_col)
  n_per   <- max(1L, floor(n_folds_subset/length(acts)))
  selected_folds <- c()
  for (a in acts) {
    idx <- which(act_col==a)
    selected_folds <- c(selected_folds, sample(idx, min(n_per,length(idx))))
  }
  rem <- setdiff(all_folds, selected_folds); short <- n_folds_subset-length(selected_folds)
  if (short>0 && length(rem)>0) selected_folds <- c(selected_folds,sample(rem,min(short,length(rem))))
  selected_folds <- sort(unique(selected_folds))
  message("Fold source  : stratified fallback (", length(selected_folds), " folds)")
  message("Fold IDs     : ", paste(selected_folds, collapse=","))

} else {
  selected_folds <- all_folds
  message("Fold source  : ALL folds (", length(selected_folds), ")")
}

# Write fold IDs used into the combo directory for traceability
writeLines(paste(selected_folds, collapse=","),
           file.path(results_root, "fold_ids_used.txt"))

# =============================================================================
# Global summary table
# =============================================================================
global_summary <- data.table(
  Repetition=integer(), Percentage=character(), Fold=integer(),
  TrainingTime_s=numeric(), SamplesRemoved=integer(), TrainSampleSize=integer(),
  MMPC_test=character(), MMPC_alpha=numeric(), MMPC_max_sx=integer(),
  LC_neighbors=character(), Action_in_LC_neighbors=logical(),
  Skeleton_connected=logical(), Skeleton_components=integer(), Skeleton_edges=integer()
)

# =============================================================================
# Helper: write a .pl file from a fitted BN
# =============================================================================
write_pl <- function(bn_fit, path) {
  of <- file(path,"w")
  writeLines(c("%%% Exogenous variables",""), con=of)
  u_idx <- 0L

  # Root nodes (exogenous)
  for (rv in bn_fit) {
    nm<-rv$node; vals<-dimnames(rv$prob)[[1]]; nv<-length(vals)
    if (!identical(rv$parents,character(0))) next
    d2<-as.data.frame(rv$prob); d2[]<-lapply(d2,as.character)
    d2$Freq<-adjust_values(as.numeric(d2$Freq))
    if (nv==2L && identical(tolower(vals),c("false","true"))) {
      u_idx<-u_idx+1L
      writeLines(paste0(format(round(d2[2,"Freq"],dec_round),nsmall=dec_round,scientific=FALSE),
                        "::u",u_idx,".\n",nm," :- u",u_idx,".\n"), con=of)
    } else {
      un<-paste0("u_",nm)
      ps<-sapply(seq_len(nv),function(j) format(round(d2[j,"Freq"],dec_round),nsmall=dec_round,scientific=FALSE))
      writeLines(c(paste(mapply(function(p,v)paste0(p,"::",un,"(",v,")"),ps,d2[,1]),collapse="; "),
                   paste0(nm,"(V) :- ",un,"(V).\n")), con=of)
    }
  }

  # Non-root nodes (endogenous)
  for (rv in bn_fit) {
    nm<-rv$node; vals<-dimnames(rv$prob)[[1]]; nv<-length(vals)
    if (identical(rv$parents,character(0))) next
    d2<-as.data.frame(rv$prob); d2[]<-lapply(d2,as.character); d2$Freq<-as.numeric(d2$Freq)
    for (ir in seq(1L,nrow(d2),by=nv)) {
      d2$Freq[ir:(ir+nv-1L)]<-adjust_values(d2$Freq[ir:(ir+nv-1L)])
      if (nv==2L && identical(tolower(vals),c("false","true"))) {
        u_idx<-u_idx+1L
        hd<-paste0(format(round(d2[ir+1L,"Freq"],dec_round),nsmall=dec_round,scientific=FALSE),"::u",u_idx,".")
        bd<-paste0(nm," :- u",u_idx)
        for (k in seq(2L,ncol(d2)-1L)) {
          cv<-unique(as.character(unlist(d2[,k]))); bd<-paste0(bd,", ")
          if (identical(tolower(cv),c("false","true"))) {
            if (identical(tolower(d2[ir,k]),"false")) bd<-paste0(bd,"\\+ ",names(d2)[k])
            else bd<-paste0(bd,names(d2)[k])
          } else bd<-paste0(bd,names(d2)[k],"(",d2[ir,k],")")
        }
        writeLines(paste0(hd,"\n",bd,".\n"), con=of)
      } else {
        u_idx<-u_idx+1L; un<-paste0("u",u_idx)
        ps<-sapply(ir:(ir+nv-1L),function(r) format(round(d2[r,"Freq"],dec_round),nsmall=dec_round,scientific=FALSE))
        writeLines(paste(mapply(function(p,v)paste0(p,"::",un,"(",v,")"),ps,vals),collapse="; "), con=of)
        bd<-paste0(nm,"(V) :- ",un,"(V)")
        for (k in seq(2L,ncol(d2)-1L)) {
          cv<-unique(as.character(unlist(d2[,k]))); bd<-paste0(bd,", ")
          if (length(cv)>2L) bd<-paste0(bd,names(d2)[k],"(",d2[ir,k],")")
          else if (identical(tolower(cv[1]),"false") && identical(tolower(cv[2]),"true")) {
            if (identical(tolower(d2[ir,k]),"false")) bd<-paste0(bd,"\\+ ",names(d2)[k])
            else bd<-paste0(bd,names(d2)[k])
          } else bd<-paste0(bd,names(d2)[k],"(",d2[ir,k],")")
        }
        writeLines(paste0(bd,".\n"), con=of)
      }
    }
  }
  close(of)
}

# =============================================================================
# Main loops
# =============================================================================
for (r_idx in seq_along(reps)) {
  rep_num <- reps[r_idx]; set.seed(seeds[r_idx])
  message("\n========== Rep ",rep_num," | ",mmpc_test," | alpha=",mmpc_alpha," | max.sx=",mmpc_max_sx," ==========")

  rep_dir      <- file.path(results_root, sprintf("rep_%d",rep_num))
  rep_test_dir <- file.path(rep_dir,"test_data")
  if (!dir.exists(rep_dir))      dir.create(rep_dir,      recursive=TRUE)
  if (!dir.exists(rep_test_dir)) dir.create(rep_test_dir, recursive=TRUE)

  dt_local    <- copy(dt)
  dt_unique_l <- copy(dt_unique)

  for (percentage in percentages) {
    message("\n--- Pct ",percentage," ---")
    cbns_dir <- file.path(rep_dir, percentage, "cBNs")
    for (d in c(file.path(rep_dir,percentage,"training_data"), cbns_dir))
      if (!dir.exists(d)) dir.create(d, recursive=TRUE)

    fraction <- as.numeric(percentage)/100L

    # Per-percentage accumulators
    tt<-numeric(); sr<-numeric(); ts<-numeric()
    lc_list<-character(); act_list<-logical()
    conn_list<-logical(); comp_list<-integer(); edge_list<-integer()

    for (fold_i in selected_folds) {
      fi <- which(selected_folds==fold_i)
      message("  Fold ",fold_i," (",fi,"/",length(selected_folds),
              ") [",mmpc_test,"|a=",mmpc_alpha,"|sx=",mmpc_max_sx,"]")

      # Test dataset
      cur_ex  <- dt_unique_l[fold_i,]
      act_lst <- c("change_to_left","change_to_right","cruise","keep","swerve_left","swerve_right")
      dt_test <- cur_ex[rep(1:.N, each=6L)]
      dt_test[, iaction := act_lst]   # act_lst has exactly 6 elements, dt_test has 6 rows
      fwrite(dt_test[,lapply(.SD,as.character),.SDcols=setdiff(names(dt_test),"state_id")],
             file.path(rep_test_dir,paste0("test_fold_",fold_i,".csv")), quote=TRUE)

      # Training dataset
      sid       <- dt_unique_l$state_id[fold_i]
      n_removed <- sum(dt_local$state_id==sid); sr<-c(sr,n_removed)
      train_dt  <- dt_local[state_id!=sid]

      ss <- round(fraction*nrow(train_dt)); au <- unique(train_dt$action)
      if (ss < length(au)) {
        train_samp <- train_dt[sample(.N,min(ss,.N))]
      } else {
        mins <- train_dt[,.SD[sample(.N,1L)],by=action]
        rem  <- ss-nrow(mins)
        train_samp <- if(rem>0) rbindlist(list(mins,train_dt[sample(.N,rem)])) else mins
      }
      ts <- c(ts, nrow(train_samp))
      df_f <- as.data.frame(lapply(
        train_samp[,lapply(.SD,as.character),.SDcols=setdiff(names(train_samp),"state_id")], factor))

      # ---- MMPC ----
      t0 <- Sys.time()
      net_mmpc <- mmpc(df_f, test=mmpc_test, alpha=mmpc_alpha, max.sx=mmpc_max_sx)

      lc_nbrs    <- tryCatch(bnlearn::nbr(net_mmpc,"latent_collision"), error=function(e) character(0))
      act_in_lc  <- "action" %in% lc_nbrs
      lc_str     <- if (length(lc_nbrs)==0L) "NONE" else paste(sort(lc_nbrs),collapse=", ")
      lc_list    <- c(lc_list,lc_str); act_list<-c(act_list,act_in_lc)

      # Skeleton connectivity
      adj      <- amat(net_mmpc); sk_edges<-nrow(arcs(net_mmpc))
      nd       <- rownames(adj); nn<-length(nd)
      vis      <- rep(FALSE,nn); comps<-list()
      for (st in seq_len(nn)) {
        if (!vis[st]) {
          q<-st; comp<-integer(0)
          while(length(q)>0){u<-q[1];q<-q[-1];if(!vis[u]){vis[u]<-TRUE;comp<-c(comp,u);nb<-which(adj[u,]==1);q<-unique(c(q,nb[!vis[nb]]))}}
          comps<-c(comps,list(nd[comp]))
        }
      }
      sk_conn<-(length(comps)==1L); sk_comp<-length(comps)
      conn_list<-c(conn_list,sk_conn); comp_list<-c(comp_list,sk_comp); edge_list<-c(edge_list,sk_edges)

      message("    LC=",lc_str," | action=",act_in_lc,
              " | connected=",sk_conn," | comps=",sk_comp)

      # Blacklist + HC
      miss <- which(adj==0 & row(adj)!=col(adj), arr.ind=TRUE)
      bl   <- unique(data.frame(from=rownames(adj)[miss[,"row"]],
                                to  =colnames(adj)[miss[,"col"]],
                                stringsAsFactors=FALSE))
      net <- hc(df_f, blacklist=bl, score="bic", debug=FALSE,
                restart=0L, perturb=1L, max.iter=Inf, maxp=Inf, optimized=TRUE)
      t_struct <- as.numeric(Sys.time()-t0, units="secs")

      # Skeleton summary
      cat(c(paste0("Fold ",fold_i," (",mmpc_test,"|alpha=",mmpc_alpha,"|max.sx=",mmpc_max_sx,")"),
            paste0("LC=",lc_str,"  action_in_LC=",act_in_lc),
            paste0("edges=",sk_edges,"  connected=",sk_conn,"  comps=",sk_comp),
            paste0("final_arcs=",nrow(arcs(net)),"  time=",round(t_struct,2),"s"), ""),
          file=file.path(cbns_dir,"skeleton_summary.txt"), sep="\n", append=TRUE)

      # Save .dot
      ob <- file.path(cbns_dir, paste0("cBN_",fold_i))
      write.dot(net, file=paste0(ob,".dot"))
      try(system(paste("dot -Tps",shQuote(paste0(ob,".dot")),
                       "-o",shQuote(paste0(ob,".ps"))),intern=TRUE),silent=TRUE)

      # Parameter learning
      t1  <- Sys.time()
      bnf <- bn.fit(net, data=df_f, method="mle", replace.unidentifiable=TRUE)
      tot <- t_struct + as.numeric(Sys.time()-t1, units="secs")
      tt  <- c(tt, tot)

      write(sprintf("Fold %d | %.2fs | rm=%d | sz=%d | %s | a=%.2f | sx=%d | LC=%s",
                    fold_i,tot,n_removed,nrow(train_samp),mmpc_test,mmpc_alpha,mmpc_max_sx,lc_str),
            file=file.path(cbns_dir,"training_numeralia.txt"), append=TRUE)

      # Write .pl
      write_pl(bnf, paste0(ob,".pl"))
      rm(train_dt,train_samp,df_f,bnf,net); gc()
    } # fold loop

    # Populate global summary
    if (length(tt)>0) {
      global_summary <- rbindlist(list(global_summary, data.table(
        Repetition=rep(rep_num,length(tt)), Percentage=rep(percentage,length(tt)),
        Fold=selected_folds[seq_along(tt)],
        TrainingTime_s=tt, SamplesRemoved=as.integer(sr), TrainSampleSize=as.integer(ts),
        MMPC_test=rep(mmpc_test,length(tt)), MMPC_alpha=rep(mmpc_alpha,length(tt)),
        MMPC_max_sx=rep(mmpc_max_sx,length(tt)),
        LC_neighbors=lc_list, Action_in_LC_neighbors=act_list,
        Skeleton_connected=conn_list, Skeleton_components=comp_list, Skeleton_edges=edge_list
      )))

      n_act<-sum(act_list); n_con<-sum(conn_list); n<-length(tt)
      message(sprintf(
        "\n  [Pct=%s] action_in_LC=%d/%d(%.1f%%) | connected=%d/%d(%.1f%%) | avg_comps=%.2f | avg_time=%.1fs",
        percentage, n_act,n,100*n_act/n, n_con,n,100*n_con/n, mean(comp_list), mean(tt)))
    }
  } # percentage loop

  # Per-rep summary
  rs <- global_summary[Repetition==rep_num]
  if (nrow(rs)>0) {
    message(sprintf("\nRep %d summary: action_in_LC=%d/%d(%.1f%%) | avg_time=%.1fs",
                    rep_num, sum(rs$Action_in_LC_neighbors), nrow(rs),
                    100*mean(rs$Action_in_LC_neighbors), mean(rs$TrainingTime_s)))
  }
} # rep loop

# =============================================================================
# Save global summary CSV
# =============================================================================
csv_path <- file.path(results_root, "global_summary.csv")
fwrite(global_summary, csv_path)
message("\nCSV: ", csv_path)

# =============================================================================
# Final summary — broken down by percentage
# =============================================================================
if (nrow(global_summary)>0) {
  message("\n=== FINAL SUMMARY: ", combo_label, " ===")
  summ <- global_summary[, .(
    n_folds          = .N,
    action_in_LC_n   = sum(Action_in_LC_neighbors),
    action_in_LC_pct = round(100*mean(Action_in_LC_neighbors),1),
    connected_n      = sum(Skeleton_connected),
    connected_pct    = round(100*mean(Skeleton_connected),1),
    avg_components   = round(mean(Skeleton_components),2),
    avg_skel_edges   = round(mean(Skeleton_edges),1),
    avg_time_s       = round(mean(TrainingTime_s),2),
    sd_time_s        = round(sd(TrainingTime_s),2)
  ), by=.(Percentage)]
  setorder(summ, Percentage)
  message(capture.output(print(summ)), sep="\n")

  # Top LC patterns per percentage
  message("\nTop LC neighbor patterns per training %:")
  for (pct in sort(unique(global_summary$Percentage))) {
    sub <- global_summary[Percentage==pct]
    lf  <- sort(table(sub$LC_neighbors), decreasing=TRUE)
    message("  Pct=",pct,":")
    for (j in seq_len(min(4L,length(lf))))
      message("    ",names(lf)[j]," : ",lf[j]," folds (",round(100*lf[j]/nrow(sub),1),"%)")
  }
}
message("\nScript finished.")
