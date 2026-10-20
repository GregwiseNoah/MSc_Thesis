import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import astropy.units as u
import sys
# sys.path.append('/home/george/Documents/Code')
# import helper_functions.Illustris_functions as func
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.rcParams.update({
'text.usetex': True,
'font.family': 'serif',
    })
time_conversion_factor = 3.08568e21 / 100000 * u.s

self_fid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/Fid_self_regulated/output_self_regulated/'
self_lowgas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/low_gas_jet/output_self/'
# self_heavyjet_path = '/lustre/rainer/scratch/20260804_061_jet_selfreg_dense/output_densejet/'


self_fid_files = sorted(glob.glob(self_fid_path+'*.hdf5'))
self_lowgas_files = sorted(glob.glob(self_lowgas_path   +'*.hdf5'))
# self_heavyjet_files = sorted(glob.glob(self_heavyjet_path   +'*.hdf5'))


final_color = '#C73E1D'
rgb = mcolors.to_rgb(final_color)
dark  = tuple(c * 0.1 for c in rgb)        
light = tuple(c + (1 - c) * 0.55 for c in rgb) 
cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', [dark, final_color, light])


def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    # print(f'Coords: {Coords.min()}, {Coords.max()}; {Coords.shape}; radii: max:{radii.max()}, min:{radii.min()}')
    return radii

def Temp_calc(data):
    
    u_cgs = np.array(data['PartType0']['InternalEnergy'], dtype = np.float64) * 10000000000
    k_B = 1.3807e-16 
    m_p = 1.6726e-24 
    mu =  .59
    gamma = 5/3

    T = u_cgs * (gamma - 1) * mu * m_p / k_B
    return T

def virial_temp_calc(data, circ_vel):
    k_B = 1.3807e-16 
    m_p = 1.6726e-24 
    mu =  .59
    G = 6.674e-8

    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    mass          = np.array(data['PartType0']['Masses']) * 1.989e+43*1/h
    # u = np.array(data['PartType0']['InternalEnergy']) * 10000000000
    vel           = np.array(data['PartType0']['Velocities']) * 100000
    bulk_velocity = np.average(vel, weights=mass, axis=0)
    vel_rel       = vel - bulk_velocity 
    # print((np.average(vel_rel, weights = mass, axis = 0))
    radii         = radii_calc(data)
    radius_trun   = data['Config'].attrs['ISO_R200'] * 3.085678e+21 *1/h
    # print(np.max(radii/3.085678e+21))
    # radius_trun = np.max(radii)
    # radius_trun = 100 * 3.085678e+21

    mask = radii < radius_trun
    mass = mass[mask]
    vel_rel = vel_rel[mask]

    
    # print("vel_disp:",np.mean(np.average(vel_rel, weights = mass, axis = 0)))
    vel_rel = np.linalg.norm(vel_rel, axis=1)
    K =  0.5 * mass * vel_rel**2
    # K =  0.5 * mass * (8e7)**2
    
    U = -2 * np.sum(K)

    M = np.sqrt(-U * radius_trun / G)
    # M = 1e14 *  1.989e+33
    # circ_vel = np.sqrt(G * M / radius_trun)
    
    virial_temp = mu * m_p * circ_vel**2 / (2*k_B) 
    # print("Mass of halo:",M, "Virial Temp:", virial_temp, "circ_vel", circ_vel /1e3)
    # print(f"virial temp = {virial_temp:.2e}")
    # return np.ones(len(radii)) * virial_temp, radii
    return virial_temp
    
def vol_avg_rho(data, nbins = 30):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    mass = np.array(data['PartType0']['Masses'], dtype = np.float64) * 1.989e+43 *1/h
    density = np.array(data['PartType0']['Density'], dtype = np.float64) * 6.769911178294544e-22 *h**2
    radii = radii_calc(data)
    volume = mass / density
    # print(np.max(radii))
    # mask = radii < 6.085678e+23
    mask = radii<r200
    radii = radii[mask]
    mass = mass[mask]
    volume = volume[mask]
    # print(len(mask))
    
    r_min, r_max = np.min(radii), np.max(radii)
    radii_bins = np.linspace(r_min, r_max, nbins + 1)

    mass_bins, _, _ = binned_statistic(radii, mass, statistic=np.sum, bins = radii_bins)
    vol_bins, _, _ = binned_statistic(radii, volume, statistic=np.sum, bins = radii_bins)

    density_bins = mass_bins/vol_bins

    radii_bin_centers = 0.5 * (radii_bins[:-1] + radii_bins[1:])

    return density_bins, ((radii_bin_centers * u.cm).to(u.kpc)).value
def vol_avg_press(data, nbins = 30):
    
    gamma = 5/3
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    
    Eint = np.array(data['PartType0']['InternalEnergy'], dtype=np.float64) * 10000000000
    mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 *1/h
    density = np.array(data['PartType0']['Density'], dtype=np.float64) * 6.769911178294544e-22 *h**2
    pressure = (gamma - 1) * density * Eint
    volume = mass / density
    radii = radii_calc(data) 

    # radius mask till 200 kpc
    # mask = radii < 6.085678e+23
    mask = radii<r200
    radii = radii[mask]
    pressure = pressure[mask]
    volume = volume[mask]
    
    r_min, r_max = np.min(radii), np.max(radii)
    radii_bins = np.linspace(r_min, r_max, nbins + 1)
    
    pvol_bins, _, _ = binned_statistic(radii, pressure * volume, statistic=np.sum, bins=radii_bins)
    vol_bins, _, _ = binned_statistic(radii, volume, statistic=np.sum, bins=radii_bins)

    # Volume-averaged pressure
    pressure_bins = pvol_bins / vol_bins

    # Bin centers
    radii_bin_centers = 0.5 * (radii_bins[:-1] + radii_bins[1:])

    return pressure_bins, ((radii_bin_centers * u.cm).to(u.kpc)).value

def mass_avg_temp(data, nbins = 30):
    Temp = Temp_calc(data)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    mass = np.array(data['PartType0']['Masses'], dtype = np.float64) * 1.989e+43 *1/h

    radii = radii_calc(data) 

    # radius mask till 200 kpc
    # mask = radii < 6.085678e+23
    mask = radii<r200
    radii = radii[mask]
    Temp = Temp[mask]
    mass = mass[mask]
    
    r_min, r_max = np.min(radii), np.max(radii)
    radii_bins = np.linspace(r_min, r_max, nbins + 1)

    masstemp_bins, _, _ = binned_statistic(radii, mass*Temp, statistic=np.sum, bins=radii_bins)
    mass_bins, _, _ = binned_statistic(radii, mass, statistic=np.sum,bins=radii_bins)

    Temp_bins = masstemp_bins / mass_bins

    radii_bin_centers = 0.5 * (radii_bins[:-1] + radii_bins[1:])

    return Temp_bins, ((radii_bin_centers * u.cm).to(u.kpc)).value

def mass_avg_vel(data, nbins = 30):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    vel = np.array(data['PartType0']['Velocities']) * 100000
    mass = np.array(data['PartType0']['Masses']) * 1.989e+43 *1/h

    radii = radii_calc(data) 

    bulk_velocity = np.average(vel, weights=mass, axis=0)
    vel_rel       = vel - bulk_velocity 
           
    # radius mask till 200 kpc
    # mask = radii < 6.085678e+23
    mask = radii<r200
    radii = radii[mask]
    vel_rel = vel_rel[mask]
    vel_rel = np.linalg.norm(vel_rel, axis=1)
    mass = mass[mask]
    
    r_min, r_max = np.min(radii), np.max(radii)
    radii_bins = np.linspace(r_min, r_max, nbins + 1)

    massvel_bins, _, _ = binned_statistic(radii, np.sqrt(mass)*vel_rel, statistic=np.sum, bins=radii_bins)
    mass_bins, _, _ = binned_statistic(radii, np.sqrt(mass), statistic=np.sum,bins=radii_bins)

    vel_rel_bins = massvel_bins/mass_bins

    radii_bin_centers = 0.5 * (radii_bins[:-1] + radii_bins[1:])

    return vel_rel_bins, ((radii_bin_centers * u.cm).to(u.kpc)).value

def r200_calc(data):
    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 *1/h # Now in gm
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) *1e2 * 1e5 / 3.086e24
    # R_vir not in config  
    # From page 45, equation for r200/rvir
    G = 6.674e-8 
    r200 = (G * m200 / (100*H**2))**(1/3) * u.cm

    return r200.value


def plot():

    plot_rows, plot_cols = 4,2
    fig, ax = plt.subplots(plot_rows, plot_cols, figsize=(4 * plot_cols + 1, 3 * plot_rows), sharex=True, sharey='row', constrained_layout=True)
    max_snapshots = 95
    n_snaps = 10  
    snapshot_numbers = np.array(np.linspace(0,1,n_snaps)*max_snapshots, dtype=np.int32)

    # cmap = cmr.ember
    # # colors = cmap(np.linspace(0.15, 0.85, n_snaps))

    # sub_cmap = mcolors.LinearSegmentedColormap.from_list(
    #                 'tropical_sub', cmr.ember(np.linspace(0, .85, 256)))
    
    colors = cmap(np.linspace(0, 1, n_snaps))
    
    # files_list = [self_fid_files, self_heavyjet_files, self_lowgas_files]
    files_list = [self_fid_files, self_lowgas_files]
    times_myr = np.full(n_snaps, np.nan)

    for i in range(plot_cols):
        files = files_list[i]
        
        for snapshot_number in tqdm(snapshot_numbers):
            with h5py.File(files[snapshot_number], 'r') as data:
                
                global r200
                r200 = r200_calc(data)
                # print(((r200 * u.cm).to(u.kpc)).value)
                rho, r_rho = vol_avg_rho(data)
                press, r_press = vol_avg_press(data)
                temp, r_temp = mass_avg_temp(data)
                vel, r_vel = mass_avg_vel(data)
                h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
                time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h

                if i == 0:  # grab time values once, from the first simulation
                    times_myr[np.searchsorted(snapshot_numbers, snapshot_number)] = time.value if hasattr(time, 'value') else time

            ax[0,i].plot(r_rho, rho, color = colors[np.searchsorted(snapshot_numbers, snapshot_number)])
            ax[1,i].plot(r_press, press, color = colors[np.searchsorted(snapshot_numbers, snapshot_number)])
            ax[2,i].plot(r_temp, temp, color = colors[np.searchsorted(snapshot_numbers, snapshot_number)])
            ax[3,i].plot(r_vel, vel*1e-5, color = colors[np.searchsorted(snapshot_numbers, snapshot_number)])

            ax[0,i].set_yscale('log')   #why did i do it like this, just use another for loop
            ax[1,i].set_yscale('log')
            ax[2,i].set_yscale('log')
            ax[3,i].set_yscale('log')
            ax[0,i].set_xscale('log')
            ax[1,i].set_xscale('log')
            ax[2,i].set_xscale('log')
            ax[3,i].set_xscale('log')
            ax[3,i].set_xlabel('r [kpc]')

            # ax[0,0].set_ylabel(r'$\frac{\sum \rho V}{\sum V}$ [g cm$^{-3}$]')
            # ax[1,0].set_ylabel(r'$\frac{\sum P V}{\sum V}$ [dyn cm$^{-2}$]')
            # ax[2,0].set_ylabel(r'$\frac{\sum M T}{\sum M}$ [K]')
            # ax[3,0].set_ylabel(r'$\frac{\sum M |v|}{\sum M}$ [km s$^{-1}$]')

            ax[0,0].set_ylabel(r'$\rho$ [g cm$^{-3}$]')
            ax[1,0].set_ylabel(r'$P$ [dyn cm$^{-2}$]')
            ax[2,0].set_ylabel(r'$T$ [K]')
            ax[3,0].set_ylabel(r'$|v|$ [km s$^{-1}$]')

            
    ax[0,0].set_title('Fiducial Jet')
    # ax[0,1].set_title('Heavy Jet')
    ax[0,1].set_title('Lower gas fraction')
    norm = mcolors.Normalize(vmin=times_myr.min(), vmax=times_myr.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02, aspect=30)
    cbar.set_label('Time [Myr]')
    # fig.tight_layout()
    plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/self_comp.png', dpi = 300, bbox_inches='tight')

if __name__=='__main__':
    plot()
