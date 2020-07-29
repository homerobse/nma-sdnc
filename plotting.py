import matplotlib.pyplot as plt
import numpy as np

from parameters import HCP_DIR


def plot_X(X, custom_title=None, vmin=None, vmax=None):
    print(X.shape)
    plt.figure()
    plt.pcolormesh(X, vmin=vmin, vmax=vmax)
    plt.colorbar(label= "BOLD")
    plt.ylabel('samples (subj,task)')
    plt.xlabel('parcels')
    if not custom_title:
        plt.title('X')
    else:
        plt.title(custom_title)


def plot_avg_bold(subj, X):
    plt.figure()
    ax=plt.plot(X[subj,:], label=f'subj {subj}') # 39 frames

    plt.title(f'Subj {subj} avg activity for 2bk_faces')
    plt.xlabel('Parcel')
    plt.ylabel('BOLD Activation(au)')
    plt.legend()

    plt.show()

def plot_cross_validation_boxplot(accuracies, kfold):
    f, ax = plt.subplots(figsize=(8, 3))
    ax.boxplot(accuracies, vert=False, widths=.7)
    ax.scatter(accuracies, np.ones(kfold))
    ax.set(
      xlabel="Accuracy",
      yticks=[],
      title=f"Average test accuracy: {accuracies.mean():.2%}"
    )
    ax.spines["left"].set_visible(False)
    print(accuracies)


from nilearn import plotting, datasets

def plot_brain_visualization(parcels_bold):
    #TODO: THIS WAS NOT TESTED, IT PROBABLY DOES NOT WORK
    with np.load(f"{HCP_DIR}/atlas.npz") as dobj:
        atlas = dict(**dobj)
    fsaverage = datasets.fetch_surf_fsaverage()
    surf_contrast = parcels_bold[atlas["labels_L"]]
    plotting.view_surf(fsaverage['infl_left'],
                       surf_contrast,
                       vmax=15)