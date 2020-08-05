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

# ts_wm_subjs = load_subjects_timeseries(remove_fixation=False)


# faces, tools, places, body
conditions_all = ['2bk_faces', '0bk_faces', '2bk_tools', '0bk_tools', '2bk_places', '0bk_places', '2bk_body', '0bk_body']
X_run0 = build_logistic_matrix(run=0, conditions=conditions_all)
y_run0 = np.concatenate(([0]*N_SUBJECTS*2,[1]*N_SUBJECTS*2, [2]*N_SUBJECTS*2, [3]*N_SUBJECTS*2))

X_run1 = build_logistic_matrix(run=1, conditions=conditions_all)
y_run1 = np.copy(y_run0)

X_run0_train, X_run0_test, y_run0_train, y_run0_test  = train_test_split(X_run0, y_run0, test_size=.2)
X_run1_train, X_run1_test, y_run1_train, y_run1_test  = train_test_split(X_run1, y_run1, test_size=.2)

X_mix_train = np.concatenate([X_run0_train, X_run1_train])
y_mix_train = np.concatenate([y_run0_train, y_run1_train])

X_mix_test = np.concatenate([X_run0_test, X_run1_test])
y_mix_test = np.concatenate([y_run0_test, y_run1_test])

print("Finished separating BOLD signal using the task blocks.")


# X_train, X_test, y_train, y_test  = train_test_split(X_run0, y_run0, test_size=.2)

#%%
print("Start regressions.")
# # First define the model
log_reg_run0 = LogisticRegression(penalty="none", max_iter=5000)
log_reg_run1 = LogisticRegression(penalty="none", max_iter=5000)
log_reg_mix = LogisticRegression(penalty="none", max_iter=5000)

# log_reg_run0 = LogisticRegression(penalty="l1", C=1, solver="saga", max_iter=5000)
# log_reg_run1 = LogisticRegression(penalty="l1", C=1, solver="saga", max_iter=5000)
# log_reg_mix = LogisticRegression(penalty="l1", C=1, solver="saga", max_iter=5000)

#Then fit it to data
log_reg_run0.fit(X_run0_train, y_run0_train)
log_reg_run1.fit(X_run1_train, y_run1_train)
log_reg_mix.fit(X_mix_train, y_mix_train)

# calculate accuracies
acc_train_run0_test_run0 = np.mean(log_reg_run0.predict(X_run0_test) == y_run0_test)
acc_train_run0_test_run1 = np.mean(log_reg_run0.predict(X_run1_test) == y_run1_test)
acc_train_run0_test_mix = np.mean(log_reg_run0.predict(X_mix_test) == y_mix_test)

acc_train_run1_test_run0 = np.mean(log_reg_run1.predict(X_run0_test) == y_run0_test)
acc_train_run1_test_run1 = np.mean(log_reg_run1.predict(X_run1_test) == y_run1_test)
acc_train_run1_test_mix = np.mean(log_reg_run1.predict(X_mix_test) == y_mix_test)

acc_train_mix_test_run0 = np.mean(log_reg_mix.predict(X_run0_test) == y_run0_test)
acc_train_mix_test_run1 = np.mean(log_reg_mix.predict(X_run1_test) == y_run1_test)
acc_train_mix_test_mix = np.mean(log_reg_mix.predict(X_mix_test) == y_mix_test)

plot_compared_accuracies(acc_train_run0_test_run0, acc_train_run0_test_run1, acc_train_run0_test_mix, acc_train_run1_test_run0, acc_train_run1_test_run1,
                             acc_train_run1_test_mix, acc_train_mix_test_run0, acc_train_mix_test_run1, acc_train_mix_test_mix)

#%%
coefs = np.copy(log_reg_run0.coef_)

categories = ["Faces", "Tools", "Places", "Body"]
for cat in range(len(categories)):
    n_most_active = 10
    most_active_regions = np.argsort(-coefs[cat,:])[:n_most_active]  # get n_most_active highest coefficients
    h = HCPRegions()
    print("\n###", categories[cat], "\n")
    for region_id in most_active_regions:
        region_name = region_info['name'][region_id]
        splithemis = region_name.split("_")[0]
        splithemis = "Right " if splithemis == "R" else "Left "
        splitreg = region_name.split("_")[1]
        print(splitreg, splithemis + h.get_entry(splitreg)["AreaDescription"], coefs[cat, region_id], region_info['network'][region_id])
