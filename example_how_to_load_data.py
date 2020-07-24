import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from parameters import *
from utils import load_timeseries, get_condition_bold, download_data

# Download the data to a folder called "hcp" in the same folder as the src folder and then run the command below.

regions = np.load(f"{HCP_DIR}/regions.npy").T
region_info = dict(
    name=regions[0].tolist(),
    network=regions[1],
    myelin=regions[2].astype(np.float),
)
print("region_info var:\n",region_info)

timeseries = load_timeseries(subject=0, task="wm", runs=1)
print("timeseries shape:\n", timeseries.shape)  # n_parcel x n_timepoint


# 2 runs, 360 parcels, 405 time-points (810 for 2 runs), per participant

# change subj and roito plot different heatmaps
subj=100
roi = 0
run = 0

t = load_timeseries(subj, 'wm', concat=False, remove_mean=True)
t = t[run]

plt.figure()

plt.subplot(1,2,1)
plt.pcolormesh(t[:,:])
plt.colorbar()
plt.title(f'Subj {subj} Activation')
plt.xlabel('Time')
plt.ylabel('Region-Id')

plt.subplot(1,2,2)
plt.title(f'{region_info["name"][roi]} Histogram')
plt.hist(t[roi]) # 0 or 1 for run, 0-360 for region

plt.show()


subj = 0
ts_wm = load_timeseries(subj, 'wm', concat=True, remove_mean=True)

condition_face = '0bk_faces'
condition_tool = '0bk_tools'
run=0
ts_avg_faces = get_condition_bold(subj,'wm', condition_face, run, ts_wm)
ts_avg_tools = get_condition_bold(subj,'wm', condition_tool, run, ts_wm)

plt.figure()

ax1=plt.plot(ts_avg_faces, label='faces') # 39 frames
ax2=plt.plot(ts_avg_tools, label='tools') # 39 frames

plt.title(f'Subj {subj} avg activity for %s' % condition_face)
plt.xlabel('Parcel')
plt.ylabel('BOLD Activation(au)')
plt.legend()

plt.show()