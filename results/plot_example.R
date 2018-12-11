raincloud_plotter("pca/extraTree_500pcs_preds_herring.npy", 
                  "../data/herring2017_celltype.csv.gz", 
                  "pca/extraTree_500pcs_preds_herring_rc.pdf", 
                  title = "Prediction on Herring 2017, extraTree trained on 
                  first 500 PCs")
raincloud_plotter("pca/extraTree_500pcs_preds_joost.npy", 
                  "../data/joost2016_celltype.csv.gz", 
                  "pca/extraTree_500pcs_preds_joost_rc.pdf", 
                  title = "Prediction on Joost 2016, extraTree trained on 
                  first 500 PCs")
