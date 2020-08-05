from parameters import *
import numpy as np
import pandas as pd
import os
import pickle

def download_data():
    # TODO: I think this is not working well, but it is not that important... We can download it manually

    if not os.path.isdir(HCP_DIR):
        os.mkdir(HCP_DIR)

    # fname = "hcp_task.tgz"
    # if not os.path.exists(fname):
    #     os.system(f"wget - qO {fname} https://osf.io/s4h8j/download/")
    #     os.system(f"tar - xzf {fname} - C {HCP_DIR} - -strip - components = 1")

    fname = "hcp_covariates.tgz"
    if not os.path.exists(fname):
        os.system(f"wget - qO {fname} https://osf.io/x5p4g/download/")
        os.system(f"tar - xzf {fname} - C {HCP_DIR} - -strip - components = 1")


def get_image_ids(task):
    """Get the image indices (1-based) for runs in a given experiment.

      Args:
        task (str) : Task of experiment ("rest" or name of task) to load
      Returns:
        run_ids (list of int) : Numeric ID for experiment image files
    """
    run_ids = [
        i for i, code in enumerate(BOLD_NAMES, 1) if task.upper() in code
    ]
    if not run_ids:
        raise ValueError(f"Found no data for '{task}''")
    return run_ids


def load_timeseries(subject, task, runs=None, concat=True, remove_mean=True, remove_fixation=False):
    """Load timeseries data for a single subject.

    Args:
      subject (int): 0-based subject ID to load
      task (str) : Task of experiment ("rest" or name of task, e.g., "wm") to load
      run (None or int or list of ints): 0-based run(s) of the task to load,
        or None to load all runs. (0 is RL, 1 is LR, this comes from the order in BOLD_NAMES variable)
      concat (bool) : If True, concatenate multiple runs in time
      remove_mean (bool) : If True, subtract the parcel-wise mean

    Returns
      ts (n_parcel x n_tp array): Array of BOLD data values
    """
    # Get the list relative 0-based index of runs to use
    if runs is None:
        runs = range(N_RUNS_REST) if task == "rest" else range(N_RUNS_TASK)
    elif isinstance(runs, int):
        runs = [runs]

    # Get the first (1-based) run id for this experiment
    offset = get_image_ids(task)[0]

    # Load each run's data
    if remove_mean and remove_fixation:
        raise ValueError("Only one can be true among remove_mean and remove_fixation, not both.")
    else:
        bold_data = [
            load_single_timeseries(subject, offset + run, remove_mean, remove_fixation) for run in runs
        ]

    # Optionally concatenate in time
    if concat:
        bold_data = np.concatenate(bold_data, axis=-1)

    return bold_data


def load_single_timeseries(subject, bold_run, remove_mean=True, remove_fixation=False):
    """Load timeseries data for a single subject and single run.

    Args:
      subject (int): 0-based subject ID to load
      bold_run (int): 1-based run index, across all tasks
      remove_mean (bool): If True, subtract the parcel-wise mean
    Returns
      ts (n_parcel x n_timepoint array): Array of BOLD data values
    """

    bold_path = f"{HCP_DIR}/subjects/{subject}/timeseries"
    bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
    ts = np.load(f"{bold_path}/{bold_file}")
    n_frames = ts.shape[1]

    if remove_mean and remove_fixation:
        raise ValueError("Only one can be true among remove_mean and remove_fixation, not both.")
    elif remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
    elif remove_fixation:
        frames = get_fixation_frames(subject)
        fixation_bold = np.array([ts[:, i] for i in frames])
        avg_fixation = np.mean(fixation_bold, axis=0, keepdims=True)

        ts -= np.tile(avg_fixation.T, n_frames)

    return ts


def load_evs(subject, name, condition):
    """Load EV (explanatory variable) data for one task condition.

    Args:
      subject (int): 0-based subject ID to load
      name (str) : Name of task
      condition (str) : Name of condition
    Returns
      evs (list of dicts): A dictionary with the onset, duration, and amplitude
        of the condition for each run.
    """
    evs = []
    for id in get_image_ids(name):
        task_key = BOLD_NAMES[id - 1]
        ev_file = f"{HCP_DIR}/subjects/{subject}/EVs/{task_key}/{condition}.txt"
        ev = dict(zip(["onset", "duration", "amplitude"], np.genfromtxt(ev_file).T))
        evs.append(ev)
    return evs


def condition_frames(run_evs, skip=0):
    """Identify timepoints corresponding to a given condition in each run.

    Args:
      run_evs (list of dicts) : Onset and duration of the event, for each run
      skip (int) : Ignore this many frames at the start of each trial, to account
        for hemodynamic lag
    Returns:
      frames_list (list of 1D arrays): List of flat arrays of frame indices, one per run
    """
    frames_list = []
    for ev in run_evs:  # loop through runs
        if not bool(ev):  # empty EV file (e.g. when there were no error trials or no no-response trials)
            frames_list.append(np.array([]))
        else:
            # Determine when trial starts, rounded down
            start = np.floor(ev["onset"] / TR).astype(int)

            # Use trial duration to determine how many frames to include for trial
            duration = np.ceil(ev["duration"] / TR).astype(int)

            # Take the range of frames that correspond to this specific trial
            if type(start) == np.ndarray:  # many trials
                frames = [s + np.arange(skip, d) for s, d in zip(start, duration)]  # loop through different onsets for each trial
            elif type(start) == float or type(start) == np.int64:
                # contains only one onset: it is either the full block (with many trials inside,
                # but just one onset value) or just a single trial
                frames = [start + np.arange(skip, duration)]

            frames_list.append(np.concatenate(frames))

    return frames_list


def get_condition_bold(subject, task, condition, run, task_bold_timeseries='', average=True):
    """
    Get BOLD signal just for the frames of a specific subject, task and condition

    Args:
    subject (int): subject id
    task (str): e.g. "wm"
    condition (str): e.g. "2bk_faces"
    run (int): id of run (0 or 1)
    task_bold_timeseries (np.ndarray size (n_parcels, n_frames)) : BOLD timesires of all frames of a task for all parcels
    average (bool): if True, returns the average bold of the condition frames
    Returns:
    condition_bold_timeseries (np.ndarray size (n_parcels,) ): average BOLD signal across block
    """
    if task_bold_timeseries == '':
        task_bold_timeseries = load_single_timeseries(subject, get_image_ids(task)[0]+run)
        
    frames = condition_frames(load_evs(subject,task,condition))[run]

    start_idx = int(frames.min())
    end_idx = int(frames.max()+1)
    if average == False:
        condition_bold_timeseries = task_bold_timeseries[:,start_idx:end_idx]
    else:
        condition_bold_timeseries = task_bold_timeseries[:,start_idx:end_idx]
        condition_bold_timeseries = np.mean(condition_bold_timeseries, axis =1)
    
    return condition_bold_timeseries


def selective_average(timeseries_data, ev, skip=0):
    """Take the temporal mean across frames for a given condition.

    Args:
      timeseries_data (array or list of arrays): n_parcel x n_tp arrays
      ev (dict or list of dicts): Condition timing information
      skip (int) : Ignore this many frames at the start of each trial, to account
        for hemodynamic lag
    Returns:
      avg_data (1D array): Data averagted across selected image frames based
      on condition timing
    """
    # Ensure that we have lists of the same length
    if not isinstance(timeseries_data, list):
        timeseries_data = [timeseries_data]
    if not isinstance(ev, list):
        ev = [ev]
    if len(timeseries_data) != len(ev):
        raise ValueError("Length of `timeseries_data` and `ev` must match.")

    # Identify the indices of relevant frames
    frames = condition_frames(ev)

    # Select the frames from each image
    selected_data = []
    for run_data, run_frames in zip(timeseries_data, frames):
        selected_data.append(run_data[:, run_frames])

    # Take the average in each parcel
    avg_data = np.concatenate(selected_data, axis=-1).mean(axis=-1)

    return avg_data


def frames_df(task, conditions):
    """ Create data frame in which each row describes the ev
    subject, run, task, condition, and frames

    Args:
    task: 'wm' in our case
    conditions: can be a single string or list of strings with which conditions to
        include in dataframe (e.g. ['0bk_faces', '2bk_faces', '0bk_err'])
    Returns:
        frames_df (pandas DataFrame): has one column with lists of frames
    """

    frames_df = pd.DataFrame([])

    for subject in subjects:
        for condition in conditions:
            evs = load_evs(subject, task, condition)
            df = pd.DataFrame(evs)  # load evs into df
            df['run'] = [0, 1]
            df['subject'] = subject
            df['condition'] = condition
            df['frames'] = condition_frames(evs)
            frames_df = frames_df.append(df, ignore_index=True)

    return frames_df


def load_subjects_timeseries(from_originals=False, remove_mean=True, remove_fixation=False):
    """ Load timeseries for all subjects into a list. Either from original files or from a pickle file

    :param from_originals: If True, will load from original files and build list
    :param remove_mean: If True, will remove mean from the whole recording
    :param remove_fixation: If True, will remove mean activity from fixation trials.
    :return: List of timeseries, one per subject
    """
    if from_originals:
        ts_wm_subjs = []
        for subj in list(subjects):
            ts_wm_subjs.append(load_timeseries(subj, 'wm', concat=True, remove_mean=remove_mean, remove_fixation=remove_fixation))
    else:
        with open(os.path.join(HCP_DIR, "ts_wm_subjs.pkl"), 'rb') as f:
            ts_wm_subjs = pickle.load(f)
    print("Subjects timeseries loaded.")
    return ts_wm_subjs


def get_fixation_frames(subject, run=0):
    """
    Get BOLD signal just for the frames of a specific subject, task and condition

    Args:
    subject (int): 0-based subject ID to load
    Returns:
    fixation_frames (): an array of frames corresponding to fixation times
    """

    trial_frames = np.append(condition_frames(load_evs(subject, 'wm', 'all_bk_cor'))[run],
                             condition_frames(load_evs(subject, 'wm', 'all_bk_err'))[run])  # TODO: include no response trials
    trial_frames = np.sort(trial_frames)

    fixation_start = np.array([], dtype=int)  # initialize

    for idx, i in enumerate(trial_frames):
        if idx == 0:
            continue

        # find frames with difference greater than 10s
        if i - trial_frames[idx - 1] > 10 / TR:
            fixation_start = np.append(fixation_start, trial_frames[idx - 1])

    fixation_duration = np.ceil(15 / TR)  # always 15s duration

    # get range of frames corresponding to duration of fixation block
    fixation_frames = np.concatenate([i + np.arange(0, fixation_duration, dtype=int) for i in fixation_start])

    return fixation_frames


def normalize_matrix(mat):
    """
    Normalize matrix elements so that they are contained in the [0,1] range

    :param mat: Matrix
    :return: normalized matrix
    """
    return (mat + abs(mat.min())) / (mat.max() - mat.min())


def build_logistic_matrix(run, conditions):
    n_conditions = len(conditions)
    X = np.empty((N_SUBJECTS * n_conditions, N_PARCELS))
    for subj in subjects:
        for k, cond in enumerate(conditions):
            X[k * N_SUBJECTS + subj, :] = get_condition_bold(subj, 'wm', cond, run)
    return X
