import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import pickle
import os

from plotting import plot_cross_validation_boxplot
from utils import *

def load_subjects_timeseries(from_originals=False):
    if from_originals:
        ts_wm_subjs = []
        for subj in list(subjects):
            ts_wm_subjs.append(load_timeseries(subj, 'wm', concat=True, remove_mean=True))
    else:
        with open(os.path.join(HCP_DIR, "ts_wm_subjs.pkl"), 'rb') as f:
            ts_wm_subjs = pickle.load(f)
    print("Subjects timeseries loaded.")
    return ts_wm_subjs

ts_wm_subjs = load_subjects_timeseries()

X_run0 = np.empty((N_SUBJECTS*4, N_PARCELS))
run = 0
for subj in list(subjects)*4:
    X_run0[subj, :] = get_condition_bold(subj,'wm','2bk_faces', run, ts_wm_subjs[subj])
    X_run0[N_SUBJECTS + subj, :] = get_condition_bold(subj,'wm','0bk_faces', run, ts_wm_subjs[subj])
    X_run0[2*N_SUBJECTS + subj, :] = get_condition_bold(subj,'wm','2bk_tools', run, ts_wm_subjs[subj])
    X_run0[3*N_SUBJECTS + subj, :] = get_condition_bold(subj,'wm','0bk_tools', run, ts_wm_subjs[subj])

X_run1 = np.empty((N_SUBJECTS*4, N_PARCELS))
run = 1
for subj in list(subjects)*4:
    X_run1[subj, :] = get_condition_bold(subj,'wm','2bk_faces', run, ts_wm_subjs[subj])
    X_run1[N_SUBJECTS + subj, :] = get_condition_bold(subj,'wm','0bk_faces', run, ts_wm_subjs[subj])
    X_run1[2*N_SUBJECTS + subj, :] = get_condition_bold(subj,'wm','2bk_tools', run, ts_wm_subjs[subj])
    X_run1[3*N_SUBJECTS + subj, :] = get_condition_bold(subj,'wm','0bk_tools', run, ts_wm_subjs[subj])

print("Finished separating BOLD signal using the task blocks.")

y_run0 = np.concatenate(([0]*N_SUBJECTS*2,[1]*N_SUBJECTS*2))
y_run1 = np.copy(y_run0)

X_run0_train, X_run0_test, y_run0_train, y_run0_test  = train_test_split(X_run0, y_run0)
X_run1_train, X_run1_test, y_run1_train, y_run1_test  = train_test_split(X_run1, y_run1)

# X = np.concatenate([X_run0_train, X_run1_train])
# y = np.concatenate([y_run0_train, y_run1_train])

X, y = X_run0, y_run0

# X_test = np.concatenate([X_run0_test, X_run1_test])
# y_test = np.concatenate([y_run0_test, y_run1_test])

#%%
print("Start regressions.")
# # First define the model
log_reg = LogisticRegression(penalty="none")

#Then fit it to data
log_reg.fit(X_run0, y_run0)

coefs = np.copy(log_reg.coef_[0,:])

n_most_active = 10
most_active_regions = np.argsort(-coefs)[:n_most_active]  # get n_most_active highest coefficients
print([(region_info['name'][region_id], coefs[region_id], region_info['network'][region_id]) for region_id in most_active_regions])

acc_train = np.mean(log_reg.predict(X_run0)==y_run0)
print("Accuracy for training set =", acc_train)
# predictions = log_reg.predict(X)
# print("Accuracy for test set =", np.mean(log_reg.predict(X_test)==y_test))


# Split across subjects
kfold=4  # k-fold cross-validation
accuracies = cross_val_score(log_reg, X_run0, y_run0, cv=kfold)

plot_cross_validation_boxplot(accuracies, kfold)

plt.show()
