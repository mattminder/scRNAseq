# Readme
## General Code Architecture
All code is present in the /src folder. The necessary datasets are in the /data folder. All generated classifiers and predictions will be generated in the /res folder.
Within the src folder, there are the subfolders helpers, network, pred, run, and train. The folder helpers contains useful tools for data loading, training and prediction, that are called by more general functions. The folder network contains the necessary tools to construct the gene network used in our network, as well as a class GeneNetworkPCA capable of doing all described network-based transformations. The pred and train folders contain functions with the prefix do_, which, when called, handle entirely a specific task, such as training a specific classifier, or prediction on multiple datasets. Finally, the run folder contains the script run.py, which, when called, will reproduce all results presented in the report.
The do_* functions don't return an output for memory utilization reasons, but save them in a specific folder within the res folder.
The entire run script takes around 6h to execute on a MacBook Pro, 2015, 8GB RAM, Intel i5 2.9 GHz

## Necessary Packages
All packages were installed with conda. 
matplotlib	3.0.1
numpy	1.15.2
pandas	0.23.4
scikit-learn	0.20.1
scipy	1.1.0
torch	1.0.0
xgboost	0.81