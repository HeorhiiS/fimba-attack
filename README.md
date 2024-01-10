# README

data_processing/ folder contains the code for data processing and data analysis of the datasets used in the paper. Please note that you would require solid storage capacity to download the whole dataset. While final dataset sizes are around a gigabyte, the intermediate files can take up a lot of storage space.

use data_download.py to download the raw files for TCGA dataset. For COVID-19 dataset please download the files from NCBI GEO database. Then use the respective processing notebooks.

models.py contains the code for all architectures and attacks as well as some evaluation code. Attacks are designed to measure confusion matrix and accuracy of the models.
shap_dl_analysis.py is the code for the SHAP analysis of the DL models and an example now how data is loaded.
runatk_standalone.py is the code standalone code for the main attack in the paper, there SHAP is computed in loop during the attack, it can be computed on either target or destination vector. This code is not used in the paper, but it is a good example of how to run the attack.

We note that the hyperparameter settings in the code can be different from the paper, and the current version was designed as a representative copy. 