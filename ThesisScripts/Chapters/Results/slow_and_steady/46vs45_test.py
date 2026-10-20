import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import glob
from tqdm import tqdm

# radius vs time and radius vs energy in


#fid resolution runs
J46_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Power_Tests/j46_t12.8/output_M13j45t128/'
J45_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Power_Tests/j45_t128/output_M13j45t128/'
# J45_t25_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Power_Tests/j45_t25/output_M13j45t128/'


J46_files = sorted(glob.glob(J46_path+"*.hdf5"))
J45_files = sorted(glob.glob(J45_path+"*.hdf5"))
# J45_t25_files = sorted(glob.glob(J45_t25_path+"*.hdf5"))

colors = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']


plt.rcParams.update({
'text.usetex': True,
'font.family': 'serif',
    })

def jet_mass_check(data):
    h = data['Header'].attrs['HubbleParam']
    gas_mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 * 1/ h   #in gm
    jet_tracer = np.array(data['PartType0']['Jet_Tracer'], dtype=np.float64) 
    # time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h

    coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) * 1 / h  # coords in kpc
    coords = coords - data['Parameters'].attrs['BoxSize']*1/h * 0.5
    axis_mask = (np.abs(coords[:, 1]) < 50) & (np.abs(coords[:, 2]) < 50)
    
    tracer_mask = jet_tracer > 1e-6
    
    jet_mask = axis_mask & tracer_mask

    jet_coords = coords[jet_mask]
    jet_radii = np.linalg.norm(jet_coords, axis=1) # will break if bh is not at center, should be fine for now
    jet_mass_tot = gas_mass[jet_mask]

    jet_mass = jet_mass_tot * jet_tracer[jet_mask]

    sorting_logic = np.argsort(jet_radii)
    sorted_radii = jet_radii[sorting_logic]
    sorted_jet_mass = jet_mass[sorting_logic]

    cumulative = np.cumsum(sorted_jet_mass)
    
    # Protecting against when jet radius is 0, such as in the 0th snapshot
    if len(cumulative) == 0:
        jet_head_radius = 0
    else:
        index  = np.argmax(cumulative >= 0.99 * cumulative[-1]) # maybe it makes more sense to compare only to the masked tracers and ignore risen jet
        jet_head_radius = sorted_radii[index]


    time_conversion_factor = 3.08568e21 / 100000 * u.s
    time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h

    return jet_head_radius* 3.085678e+21, time

def jet_files(filelist):
    jet_radii = []
    times = []

    for File in tqdm(filelist):
        with h5py.File(File, 'r') as data:
            radius, time = jet_mass_check(data)
        jet_radii.append(radius)
        times.append(time.value)
    
    return np.array(jet_radii), np.array(times)

if __name__ == '__main__':

    #E dot
    #load times to get energy

    J46_power = (1e46*(u.erg / u.s)).to(u.erg/u.Myr).value
    J45_power = (1e45*(u.erg / u.s)).to(u.erg/u.Myr).value

    J46_radii, J46_times = jet_files(J46_files)
    J45_radii, J45_times = jet_files(J45_files)
    # J45_t25_radii, J45_t25_times = jet_files(J45_t25_files)

    # J46_energies = J46_times*J46_times 
    # J45_energies = J45_times*J45_times   THIS IS WRONG, DOESN'T JET CUTOFF INTO ACCOUNT

    J46_cutoff = 12.8 #*u.Myr
    J45_cutoff = 128 #*u.Myr
    # J45_t25_cutoff = 25 # Myr

    J46_energies = np.ones(len(J46_times))*J46_cutoff*J46_power
    J46_energies[J46_times<=J46_cutoff] = J46_times[J46_times<=J46_cutoff] * J46_power

    J45_energies = np.ones(len(J45_times))*J45_cutoff*J45_power
    J45_energies[J45_times<=J45_cutoff] = J45_times[J45_times<=J45_cutoff] * J45_power

    # J45_t25_energies = np.ones(len(J45_t25_times))*J45_t25_cutoff*J45_power #J45 power still applies here
    # J45_t25_energies[J45_t25_times<=J45_t25_cutoff] = J45_t25_times[J45_t25_times<=J45_t25_cutoff] * J45_power

    fig, ax = plt.subplots(1, 2, figsize = (12,6))

    
    ax[0].plot(J46_times, (J46_radii*u.cm).to(u.kpc), color=colors[3], label =r'$P_{\mathrm{jet,\ J46}} = 10^{46} \mathrm{\ erg/s}$',marker='o')
    ax[0].plot(J45_times, (J45_radii*u.cm).to(u.kpc), color=colors[4], label =r'$P_{\mathrm{jet.\ J45}} = 10^{45} \mathrm{\ erg/s}$',marker='o')
    # ax[0].plot(J45_t25_times, (J45_t25_radii*u.cm).to(u.kpc), color=colors[2], label ='J45_t25',marker='o')

    #vertical lines for jet cutoff
    ax[0].axvline(J46_cutoff, color=colors[3], label =r'J46 jet cutoff', linestyle='--', alpha=0.6) 
    ax[0].axvline(J45_cutoff, color=colors[4], label =r'J45 jet cutoff', linestyle='--', alpha=0.6)

    ax[1].plot(J46_energies, (J46_radii*u.cm).to(u.kpc), color=colors[3],marker='o')
    ax[1].plot(J45_energies, (J45_radii*u.cm).to(u.kpc), color=colors[4],marker='o')
    # ax[1].plot(J45_t25_energies, (J45_t25_radii*u.cm).to(u.kpc), color=colors[2], label ='J45_t25',marker='o')

    ax[0].set_ylabel('Radius [kpc]')
    ax[0].set_xlabel('Time [Myr]')
    ax[1].set_xlabel('Energy (erg)')

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].legend()
    plt.tight_layout()
    plt.savefig('Plots/46vs45fid_final.png', dpi=300)

    


