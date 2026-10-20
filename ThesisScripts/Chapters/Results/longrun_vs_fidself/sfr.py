# All.Time, total_sm, totsfrrate, rate_in_msunperyear

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# import cmasher as cmr
# cmap = cmr.prinsenvlag
colors = ['#2D728F', '#AB3428']

time_conversion_factor = 3.08568e21 / 100000 * u.s
h = 0.6774


longjet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/cool_j43_t1000_1/output_longjet/sfr.txt'
selffid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/Fid_self_regulated/output_self_regulated/sfr.txt'



plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
                     })

paths = [longjet_path, selffid_path]
labels = ['Single outburst', 'Self regulated']

fig, ax = plt.subplots(figsize=(7,5))
for i, path in enumerate(paths):
    data = np.loadtxt(path)

    time = (data[:, 0] * 1/h * time_conversion_factor).to(u.Myr) 
    sfr = data[:, 2] 

    if i == 1:
        time = time[:-6500] #either bin or cut off the last few sfr values for fid self reg
        sfr = sfr[:-6500]
    ax.plot(time.value, sfr, color=colors[i], label = labels[i])


ax.set_xlabel('Time [Myr]')
ax.set_ylabel(r'SFR [$\mathrm{\ M_{\odot}/yr}$]')
ax.set_ylim(1e-2, 1e1)
# ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/fidvslong_sfr.png', dpi = 300)
