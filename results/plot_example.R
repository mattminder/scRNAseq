raincloud_plotter("base_features/log_lasso_preds_herring.npy", 
                  "../data/herring2017_celltype.csv.gz", 
                  "base_features/log_lasso_preds_herring_rc.pdf", 
                  title = "Prediction on Herring 2017, Logistic Lasso trained on 
                  untransformed data.")
raincloud_plotter("base_features/log_lasso_preds_joost.npy", 
                  "../data/joost2016_celltype.csv.gz", 
                  "base_features/log_lasso_preds_joost_rc.pdf", 
                  title = "Prediction on Joost 2016, Logistic Lasso trained on 
                  untransformed data.")
