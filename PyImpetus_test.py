# PyImpetus_test
from PyImpetus import PPIMBC, PPIMBR
import pandas as pd
import numpy as np
from sklearn.svm import SVC
# Import the algorithm. PPIMBC is for classification and PPIMBR is for regression
# Initialize the PyImpetus object
model = PPIMBC(model=SVC(random_state=27, class_weight="balanced"), p_val_thresh=0.05, num_simul=30, simul_size=0.2, simul_type=0, sig_test_type="non-parametric", cv=5, random_state=27, n_jobs=-1, verbose=2)
# The fit_transform function is a wrapper for the fit and transform functions, individually.
# The fit function finds the MB for given data while transform function provides the pruned form of the dataset
df_train = model.fit_transform(df_train.drop("Response", axis=1), df_train["Response"].values)
df_test = model.transform(df_test)
# Check out the MB
print(model.MB)
# Check out the feature importance scores for the selected feature subset
print(model.feat_imp_scores)
# Get a plot of the feature importance scores
model.feature_importance()