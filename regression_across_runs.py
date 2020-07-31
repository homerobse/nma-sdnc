import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, datasets

from hcp_regions import HCPRegions
from parameters import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import pickle
import os
from plotting import *

from plotting import plot_cross_validation_boxplot, plot_brain_visualization
from utils import *

ts_wm_subjs = load_subjects_timeseries()


# faces, tools, places, body
conditions_all = ['2bk_faces', '0bk_faces', '2bk_tools', '0bk_tools', '2bk_places', '0bk_places', '2bk_body', '0bk_body']
X_run0 = build_logistic_matrix(ts_wm_subjs, run=0, conditions=conditions_all)
y_run0 = np.concatenate(([0]*N_SUBJECTS*2,[1]*N_SUBJECTS*2, [2]*N_SUBJECTS*2, [3]*N_SUBJECTS*2))

X_run1 = build_logistic_matrix(ts_wm_subjs, run=1, conditions=conditions_all)
y_run1 = np.copy(y_run0)

print("Finished separating BOLD signal using the task blocks.")

X_train = X_run0
y_train = y_run0
X_test = np.concatenate([X_run1])
y_test = np.concatenate([y_run1])

#%%
print("Start regressions.")
# # First define the model
# log_reg = LogisticRegression(penalty="none")
log_reg = LogisticRegression(penalty="l1", C=1, solver="saga", max_iter=5000)

#Then fit it to data
log_reg.fit(X_train, y_train)

coefs = np.copy(log_reg.coef_[0,:])
#%%
n_most_active = 10
most_active_regions = np.argsort(-coefs)[:n_most_active]  # get n_most_active highest coefficients
h = HCPRegions()
for region_id in most_active_regions:
    region_name = region_info['name'][region_id]
    splithemis = region_name.split("_")[0]
    splithemis = "Right" if splithemis == "R" else "Left"
    splitreg = region_name.split("_")[1]
    print((splithemis, h.get_entry(splitreg)["AreaDescription"], coefs[region_id], region_info['network'][region_id]))


plot_train_test_accuracy(log_reg, X_train, y_train, X_test, y_test)
