# All.Time, total_sm, totsfrrate, rate_in_msunperyear

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.stats import binned_statistic
# import cmasher as cmr
# cmap = cmr.prinsenvlag
colors = ["#5b5f97","#ff6b6c", 'black']

time_conversion_factor = 3.08568e21 / 100000 * u.s
h = 0.6774

self_fid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/Fid_self_regulated/output_self_regulated/sfr.txt'
self_lowgas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/low_gas_jet/output_self/sfr.txt'
# self_heavyjet_path = '/lustre/rainer/scratch/20260804_061_jet_selfreg_dense/output_densejet/sfr.txt'

nojet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/Jet_efficiency_0/output_self_regulated/sfr.txt'

plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
                     })

# paths = [self_fid_path, self_heavyjet_path, self_lowgas_path]
# labels = ['Fiducial', 'Dense jet', 'Lower gas fraction']

paths = [self_fid_path, self_lowgas_path, nojet_path]
labels = ['Fiducial', 'Lower gas fraction', 'No jet']

fig, ax = plt.subplots(figsize=(7,5))
for i, path in enumerate(paths):
    data = np.loadtxt(path)

    time = (data[:, 0] * 1/h * time_conversion_factor).to(u.Myr) 
    time = time.value
    sfr = data[:, 2] 
    # print(time.dtype)
    time_bins = np.linspace(time.min(), time.max(), 100)
    sfr_bins,_,_ = binned_statistic(time, sfr, statistic='median', bins = time_bins)
    time_bin_centers = 0.5 * (time_bins[1:] + time_bins[:-1])

    ax.plot(time_bin_centers, sfr_bins, color=colors[i], label = labels[i])
    # zeros = np.where(sfr <= 1e-4)[0]
    # print(len(zeros))
    # print(len(sfr))

ax.set_xlabel('Time [Myr]')
ax.set_ylabel(r'SFR [$\mathrm{\ M_{\odot}/yr}$]')
# ax.set_xlim(1e1, 1e3)
# ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/self_3jet_sfr.png', dpi = 300)
