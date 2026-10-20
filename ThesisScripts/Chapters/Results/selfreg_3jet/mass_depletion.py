import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import h5py
import glob
import astropy.units as u
import sys
# import cmasher as cmr
from tqdm import tqdm


colors = ["#5b5f97","#ff6b6c"] # ,"#360a14"
#number of expected plot lines
# cmap =cmr.ember

time_conversion_factor = 3.08568e21 / 100000 * u.s
plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
                     })

self_fid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/Fid_self_regulated/output_self_regulated/'
self_lowgas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/low_gas_jet/output_self/'
# self_heavyjet_path = '/lustre/rainer/scratch/20260804_061_jet_selfreg_dense/output_densejet/'


self_fid_files = sorted(glob.glob(self_fid_path+'*.hdf5'))
self_lowgas_files = sorted(glob.glob(self_lowgas_path   +'*.hdf5'))
# self_heavyjet_files = sorted(glob.glob(self_heavyjet_path   +'*.hdf5'))


def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    # shifted_Coords = [i - [boxsize*0.5, boxsize*0.5, boxsize*0.5] for i in Coords]
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    # print(f'Coords: {Coords.min()}, {Coords.max()}; {Coords.shape}; radii: max:{radii.max()}, min:{radii.min()}')
    return radii


fig, ax = plt.subplots(figsize=(7,5))

def massinvirrad(files_list, masses_halo_Msun): #dont actually need masses_halo_Msun

    for i, files in enumerate(files_list):
        masses = []
        times = []
        mask_val = (50 * u.kpc).to(u.cm)
        for file in tqdm(files):
            with h5py.File(file, 'r') as data:
                # file = h5py.File(files[frame], 'r')
                
                h =  np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
                # coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) * 1/h
                radii = radii_calc(data)
                mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1/h *1e10 # in m_sun

                mask = radii <  mask_val.value

                radii = radii[mask]
                mass = mass[mask]

                total_mass = np.sum(mass) #/ masses_halo_Msun[i] #normalise by halo mass
                time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h

                # print(time)
                masses.append(total_mass)
                times.append(time.value)
            # print(masses[-1], times[-1])

        # print()
        ax.plot(times[:], masses[:]/masses[0],color = colors[i], label=labels[i])

    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$M_{\mathrm{enc}}(<r = 50 kpc) / M_{\mathrm{enc,\, initial}}$')
    ax.legend()
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/Massdepletion_3self.png')
                

# labels = ['Fiducial jet', 'Heavy Jet', 'Lower Gas Fraction']
# massinvirrad([self_fid_files, self_heavyjet_files, self_lowgas_files], [1e13,1e13,1e13])


labels = ['Fiducial jet', 'Lower Gas Fraction']
massinvirrad([self_fid_files, self_lowgas_files], [1e13, 1e13])