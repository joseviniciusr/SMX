# Importing the necessary libraries
import pandas as pd
import numpy as np
import kennard_stone as ks
pd.options.plotting.backend = 'plotly'  # setting plotly as the backend for pandas plotting

# Add parent directory to sys.path so local module 'synthetic' can be imported
import sys
from pathlib import Path # for path manipulations
# Move three levels up from current working directory to reach workspace root
workspace_root = Path.cwd().parent.parent.parent.resolve() 
smx_dir = workspace_root / 'smx'  # Path to smx folder
if str(smx_dir) not in sys.path: # check to avoid duplicates
    sys.path.insert(0, str(smx_dir)) # insert at the start of sys.path to prioritize local modules

# Loading a soil spectral dataset based on X-ray fluorescence (XRF)
data_complete = pd.read_csv(f'{workspace_root}/real_datasets/xrf/sweet_pepper.csv', sep=';') # local copy of Toledo 2022 dataset (os ... indica para omitir o caminho longo)
data = data_complete.loc[:, '2.12':'23.08']

# Split dataset by class and create calibration/prediction sets using Kennard-Stone (as in original pipeline)
data_A = data_complete[data_complete['Class'] == 'A'].reset_index(drop=True)
data_B = data_complete[data_complete['Class'] == 'B'].reset_index(drop=True)

# splitting the data into calibration and prediction sets by kennard-stone algorithm
XA_cal, XA_pred = ks.train_test_split(data_A.loc[:, '2.12':'23.08'], test_size=0.30)  # class A
XA_cal = XA_cal.reset_index(drop=True)
XA_pred = XA_pred.reset_index(drop=True)

XB_cal, XB_pred = ks.train_test_split(data_B.loc[:, '2.12':'23.08'], test_size=0.30)  # class B
XB_cal = XB_cal.reset_index(drop=True)
XB_pred = XB_pred.reset_index(drop=True)

Xcalclass = pd.concat([XA_cal, XB_cal], axis=0).reset_index(drop=True)  # concatenating both classes
Xpredclass = pd.concat([XA_pred, XB_pred], axis=0).reset_index(drop=True)
ycalclass = pd.Series(['A']*XA_cal.shape[0] + ['B']*XB_cal.shape[0])  # target for calibration set
ypredclass = pd.Series(['A']*XA_pred.shape[0] + ['B']*XB_pred.shape[0])  # target for prediction set

# preprocessings
import preprocessings as prepr  # preprocessing methods for XRF data

Xcalclass_prep, mean_calclass, mean_calclass_poisson  = prepr.poisson(Xcalclass, mc=True)
Xpredclass_prep = ((Xpredclass/np.sqrt(mean_calclass)) - mean_calclass_poisson)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
# PLS-DA with optimized latent variables
>>>>>>> Stashed changes
=======
# PLS-DA with optimized latent variables
>>>>>>> Stashed changes
from modeling import svm_optimized

svm_model = svm_optimized(Xcalclass_prep, ycalclass, Xpredclass_prep, ypredclass, aim='classification', kernel='rbf')

import shap

model_predict_proba = lambda x: svm_model[3].predict_proba(x)[:, 1] # o 1 é a probabilidade da classe positiva
<<<<<<< Updated upstream
<<<<<<< Updated upstream
explainer = shap.KernelExplainer(model_predict_proba, Xcalclass_prep, njobs=11)  # using a subset of calibration data as background for SHAP
=======
explainer = shap.KernelExplainer(model_predict_proba, Xcalclass_prep, njobs=-1)  # using a subset of calibration data as background for SHAP
>>>>>>> Stashed changes
=======
explainer = shap.KernelExplainer(model_predict_proba, Xcalclass_prep, njobs=-1)  # using a subset of calibration data as background for SHAP
>>>>>>> Stashed changes
shap_exp = explainer(Xcalclass_prep)  # explain a subset of calibration data

shap_values = shap_exp.values
shap_global_importance = pd.DataFrame({
    'energy': Xcalclass_prep.columns,
    'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)})
shap_global_importance.to_csv('shap_sweet_pepper.csv', index=False, sep=';')