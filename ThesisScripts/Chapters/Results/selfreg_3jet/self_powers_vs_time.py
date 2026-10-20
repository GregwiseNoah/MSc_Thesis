import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import glob
from tqdm import tqdm
import cmasher as cmr

# cmap = cmr.prinsenvlag
# colors = cmap(np.linspace(0, 1, 6)) # stop using diverging colormaps :(

colors1 = ["#5b5f97","#360a14","#ff6b6c"]


self_fid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/Fid_self_regulated/output_self_regulated/'
self_lowgas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/low_gas_jet/output_self/'
# self_heavyjet_path = '/lustre/rainer/scratch/20260804_061_jet_selfreg_dense/output_densejet/'


self_fid_files = sorted(glob.glob(self_fid_path+'*.hdf5'))
self_lowgas_files = sorted(glob.glob(self_lowgas_path   +'*.hdf5'))
# self_heavyjet_files = sorted(glob.glob(self_heavyjet_path   +'*.hdf5'))


time_conversion_factor = 3.08568e21 / 100000 * u.s
plt.rcParams.update({
'text.usetex': True,
'font.family': 'serif',
    })

def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    return radii

def r200_calc(data):
    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 *1/h # Now in gm
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) *1e2 * 1e5 / 3.086e24
    # R_vir not in config  
    # From page 45, equation for r200/rvir
    G = 6.674e-8 
    r200 = (G * m200 / (100*H**2))**(1/3) 

    return r200 #return in cgs


def jet_power_t(filelist):

    Edots = []
    times = []

    
    for File in tqdm(filelist):
        with h5py.File(File, 'r') as data:
            h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
            mdot = np.array(data['PartType5']['BH_Mdot'], dtype=np.float64) * 1/h * 6.445909132449985e+26 #now in cgs
            Edot = mdot * 0.2 * 3e10**2 # mdot * efficiency * c^2
            time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr)  * 1/h
        Edots.append(Edot)
        times.append(time.value)
    
    return np.array(Edots), np.array(times)

def cooling_lum(filelist):

    Ls = []
    times = []

    for File in tqdm(filelist):
        X_H = 0.76
        mp = 1.67377e-24 
        with h5py.File(File, 'r') as data:
            h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
            Cooling_rate_allcells = np.array(data['PartType0']['GFM_CoolingRate'], dtype=np.float64)
            density = np.array(data['PartType0']['Density'], dtype=np.float64) * h**2 * 6.769911178294544e-22 #Now in cgs
            mass = np.array(data['PartType0']['Masses'], dtype=np.float64)  * 1.989e+43  *1/h

            radii = radii_calc(data)
            r200 = r200_calc(data)

            mask = radii < r200
            radii = radii[mask]
            Cooling_rate = Cooling_rate_allcells[mask]
            density = density[mask]
            mass = mass[mask]
            
            n_h = density * X_H / mp 
            L_cells = Cooling_rate * n_h**2 * mass /  density 

            L = np.sum(L_cells)
            time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr)  * 1/h
        Ls.append(L)
        times.append(time.value)
    return np.array(Ls), np.array(times)



if __name__=='__main__':

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fid_Edots, fid_times = jet_power_t(self_fid_files)
    fid_Ls, fid_times = cooling_lum(self_fid_files)

    lowg_Edots, lowg_times = jet_power_t(self_lowgas_files)
    lowg_Ls, lowg_times = cooling_lum(self_lowgas_files)

    # densejet_Edots, densejet_times = jet_power_t(self_heavyjet_files)
    # densejet_Ls, densejet_times = cooling_lum(self_heavyjet_files)

    # plt.rcParams.update({
    #     'font.size': 11,
    #     'axes.linewidth': 1.0,
    #     'xtick.direction': 'in',
    #     'ytick.direction': 'in',
    #     'xtick.top': True,
    #     'ytick.right': True,
    # })

    fig, ax = plt.subplots(figsize=(6, 4))

    lw = 1.2
    scenarios = [
        (fid_times, fid_Edots, fid_Ls, colors1[0], 'Fiducial'),
        (lowg_times, lowg_Edots, lowg_Ls, colors1[2], 'Lower gas fraction')
        # (densejet_times, densejet_Edots, densejet_Ls, colors1[1], 'Dense jet'),
    ]

    for times, Edots, Ls, color, _ in scenarios:
        ax.plot(times, Edots, color=color, lw=lw, linestyle='-')
        ax.plot(times, -Ls, color=color, lw=lw, linestyle='--')  # L is negative

    ax.set_ylabel(r'$\dot E$ [erg/s]')
    ax.set_xlabel('Time [Myr]')
    ax.set_yscale('log')
    # ax.grid(True, which='major', alpha=0.25, linewidth=0.6)
    # ax.grid(True, which='minor', alpha=0.1, linewidth=0.4)

    # Two-part legend: color -> scenario, linestyle -> quantity
    color_handles = [Line2D([0], [0], color=c, lw=lw, label=name)
                    for *_, c, name in scenarios]
    style_handles = [
        Line2D([0], [0], color='k', lw=lw, linestyle='-', label=r'$\dot E_{\mathrm{jet}}$'),
        Line2D([0], [0], color='k', lw=lw, linestyle='--', label=r'$|L|$'),
    ]

    leg1 = ax.legend(handles=color_handles, loc='upper right', fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/self_3jet_powers_vs_time_comp.png', dpi=300, bbox_inches='tight')

    # fig, ax = plt.subplots(figsize = (6,4))


    # # fid_Edots, fid_times = jet_power_t(self_regulated_files)
    # # fid_Ls, fid_times = cooling_lum(self_regulated_files)

    # # ax.plot(fid_times, fid_Edots, color = colors1[2], label=r'$\dot E_{\mathrm{jet,\ low res}}$')
    # # ax.plot(fid_times, -fid_Ls, color = colors2[2], label=r'$|L_{\mathrm{jet,\ low res}}|$') # Putting negative here since L is negative

    # fid_Edots, fid_times = jet_power_t(self_fid_files)
    # fid_Ls, fid_times = cooling_lum(self_fid_files)

    # lowg_Edots, lowg_times = jet_power_t(self_lowgas_files)
    # lowg_Ls, lowg_times = cooling_lum(self_lowgas_files)

    # densejet_Edots, densejet_times = jet_power_t(self_heavyjet_files)
    # densejet_Ls, densejet_times = cooling_lum(self_heavyjet_files)

    
    # ax.plot(fid_times, fid_Edots, color = colors1[4], label=r'$\dot E_{\mathrm{jet,\ fid}}$')
    # ax.plot(lowg_times, lowg_Edots, color = colors1[3], label=r'$\dot E_{\mathrm{jet,\ lower\ gas\ fraction}}$')
    # ax.plot(densejet_times, densejet_Edots, color = colors1[2], label=r'$\dot E_{\mathrm{dense\ jet}}$')
    
    # ax.plot(fid_times, -fid_Ls, color = colors1[4], linestyle='--', label=r'$|L_{\mathrm{jet,\ fid}}|$') # Putting negative here since L is negative
    # ax.plot(lowg_times, -lowg_Ls, color = colors1[3], linestyle='--', label=r'$|L_{\mathrm{jet,\ lower\ gas\ fraction}}|$')
    # ax.plot(densejet_times, -densejet_Ls, color = colors1[2], linestyle='--', label=r'$|L_{\mathrm{dense\ jet}}|$')

    # ax.set_ylabel(r'$\dot \mathrm{E}$ [erg/s]')
    # ax.set_xlabel('Time [Myr]')
    # ax.set_yscale('log')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('Plots/self_3jet_powers_vs_time_comp.png', dpi = 300)
    