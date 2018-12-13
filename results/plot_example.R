library(glue)
subfolder = "nested"
method = "xgboost_nested"
title_suffix = ""
predfiletype = "csv"
col=1
thresh = NULL
raincloud_plotter(glue("{subfolder}/{method}_preds_herring.{predfiletype}"), 
                  "../data/herring2017_celltype.csv.gz", 
                  glue("{subfolder}/{method}_preds_herring_rc.pdf"), 
                  title = glue("Prediction on Herring 2017, {method} {title_suffix}"),
                  col = col,
                  thresh = thresh)
raincloud_plotter(glue("{subfolder}/{method}_preds_joost.{predfiletype}"), 
                  "../data/joost2016_celltype.csv.gz", 
                  glue("{subfolder}/{method}_preds_joost_rc.pdf"), 
                  title = glue("Prediction on Joost 2016, {method} {title_suffix}"),
                  col=col,
                  thresh = thresh
                  )
