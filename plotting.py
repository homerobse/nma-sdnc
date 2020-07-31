import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting, datasets

from parameters import atlas


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


def plot_brain_visualization(parcels_bold):
    """Plot brain activity map
    :param parcels_bold: numpy array with one Bold activity signal for each parcel
    """
    fsaverage = datasets.fetch_surf_fsaverage()
    surf_parcels_bold = parcels_bold[atlas["labels_L"]]
    plotting.view_surf(fsaverage['infl_left'],
                       surf_parcels_bold)


def plot_train_test_accuracy(log_reg, X_train, y_train, X_test, y_test):
    acc_train = np.mean(log_reg.predict(X_train)==y_train)
    acc_test = np.mean(log_reg.predict(X_test)==y_test)

    def autolabel(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('%.1f' % height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig, ax = plt.subplots()
    plt.title("Decoding accuracy: Training vs Test")
    rect = plt.bar(["Training","Test"], [100*acc_train,100*acc_test])
    plt.ylabel("Accuracy (%)")
    autolabel(rect, ax)

