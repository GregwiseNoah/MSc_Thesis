

import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cmasher as cmr
import astropy.units as u
# sys.path.append('/home/george/Documents/Code')
# import helper_functions.Illustris_functions as func
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic
from importlib import reload



e13_filespath = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Nojet/output_M13j45t128/'
e14_filespath = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Nojet/output_M13j46t25/'

e13_files = sorted(glob.glob(e13_filespath+ "snap*"))
e14_files = sorted(glob.glob(e14_filespath+ "snap*"))

cmap = cmr.prinsenvlag
colors= cmap(np.linspace(0, 1, 2))

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
    

def vol_avg_rho(data, nbins = 1000):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    mass = np.array(data['PartType0']['Masses'], dtype = np.float64) * 1.989e+43 *1/h
    density = np.array(data['PartType0']['Density'], dtype = np.float64) * 6.769911178294544e-22 *h**2
    radii = radii_calc(data)
    volume = mass / density
    # print(np.max(radii))
    mask = radii < 6.085678e+23
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

    return density_bins, radii_bin_centers 


def vol_avg_press(data, nbins = 1000):
    
    gamma = 5/3
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    
    Eint = np.array(data['PartType0']['InternalEnergy'], dtype=np.float64) * 10000000000
    mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 *1/h
    density = np.array(data['PartType0']['Density'], dtype=np.float64) * 6.769911178294544e-22 *h**2
    pressure = (gamma - 1) * density * Eint
    volume = mass / density
    radii = radii_calc(data) 

    # radius mask till 200 kpc
    mask = radii < 6.085678e+23
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

    return pressure_bins, radii_bin_centers

def mass_avg_temp(data, nbins = 1000):
    Temp = Temp_calc(data)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    mass = np.array(data['PartType0']['Masses'], dtype = np.float64) * 1.989e+43 *1/h

    radii = radii_calc(data) 

    # radius mask till 200 kpc
    mask = radii < 6.085678e+23
    radii = radii[mask]
    Temp = Temp[mask]
    mass = mass[mask]
    
    r_min, r_max = np.min(radii), np.max(radii)
    radii_bins = np.linspace(r_min, r_max, nbins + 1)

    masstemp_bins, _, _ = binned_statistic(radii, mass*Temp, statistic=np.sum, bins=radii_bins)
    mass_bins, _, _ = binned_statistic(radii, mass, statistic=np.sum,bins=radii_bins)

    Temp_bins = masstemp_bins / mass_bins

    radii_bin_centers = 0.5 * (radii_bins[:-1] + radii_bins[1:])

    return Temp_bins, radii_bin_centers

def mass_avg_vel(data, nbins = 1000):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    vel = np.array(data['PartType0']['Velocities']) * 100000
    mass = np.array(data['PartType0']['Masses']) * 1.989e+43 *1/h

    radii = radii_calc(data) 

    bulk_velocity = np.average(vel, weights=mass, axis=0)
    vel_rel       = vel - bulk_velocity 
           
    # radius mask till 200 kpc
    mask = radii < 6.085678e+23
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

    return vel_rel_bins, radii_bin_centers

def sim_data( lists_of_sims, file_flag = 1, sim_info = None):
    fig, axs = plt.subplots(2,2, figsize = (9,6
                                           ), sharex= True )
    axs = axs.flatten() 
    time_conversion_factor = 3.08568e21 / 100000 * u.s
    circ_vels = [662.981 * 100000, 662.981 * 100000, 662.981 * 100000]

    plot_info = [
          "Volume-averaged density",
          "Volume-averaged pressure",
          "Mass-averaged temperature",
          "Mass-averaged velocity"]
    
    for num, i in enumerate(lists_of_sims):
        if file_flag == 1:
            files = [i[0]]
        elif file_flag == 2:
            files = [i[-1]]
        elif file_flag == 3:
            files = [i[0], i[-1]]
        else:
            raise ValueError("file_flag must be 1, 2, or 3")
            
        for number, file in enumerate(files):
            with h5py.File(file, 'r') as data:

                # fig.suptitle('Simulation information')
                h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
                time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h
                density_bins, radii_bins = vol_avg_rho(data, 20)
                # density_median_bins, radii_median_bins = func.median_binner(density_bins, radii_bins, nbins = 100)         
                # axs[0].plot(radii_median_bins, density_median_bins, label=f'{time:.2f}') # volume averaged density
                axs[0].plot((radii_bins * u.cm).to(u.kpc), density_bins, color=colors[num],
                            label=f'{sim_info[num]}')
                
                # axs[0].plot((radii_bins * u.cm).to(u.kpc), density_bins,
                #             label=f'{sim_info[num] if sim_info and number==0 else ""}') # volume averaged density
                axs[0].set_ylabel(plot_info[0])


                axs[0].set_yscale('log')
                
                pressure_bins, radii_bins = vol_avg_press(data, 20)
                axs[1].plot((radii_bins* u.cm).to(u.kpc), pressure_bins , color=colors[num]) # volume averaged pressure
                axs[1].set_ylabel(plot_info[1])
                axs[1].set_yscale('log')

                temp_bins, radii_bins = mass_avg_temp(data, nbins = 20)
                # virial_temp = virial_temp_calc(data, circ_vels[num])
                # if number == 0: 
                #     axs[2].axhline(y=virial_temp, color='r', linestyle='dashed',
                #                    label=rf'$T_{{vir}}$') #{sim_info[num]}')
                # else:
                #     axs[2].axhline(y=virial_temp, color='r', linestyle='dashed')
                axs[2].plot((radii_bins* u.cm).to(u.kpc), temp_bins, color=colors[num]) # mass averaged temp
                axs[2].set_ylabel(plot_info[2])                
                axs[2].set_yscale('log')

        
                vel_rel_bins, radii_bin_centers = mass_avg_vel(data, nbins = 20)
                axs[3].plot((radii_bin_centers* u.cm).to(u.kpc), (vel_rel_bins*u.cm / u.s).to(u.km / u.s), color=colors[num])#, #label=f'{time:.2f}') # mass averaged absolute velocity  
                axs[3].set_ylabel(plot_info[3])
                # a
                axs[3].set_yscale('log')

    # if sim_info:
        
    axs[-1].set_xlabel("Radius (kpc)")
    axs[2].set_xlabel("Radius (kpc)")
    axs[0].legend()
    handles, labels = axs[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[2].legend(by_label.values(), by_label.keys(), loc='upper right')
    #axs[2].legend(loc='upper right')
    # axs[3].legend()
    plt.suptitle(r'Initial Conditions')
    plt.tight_layout()
    plt.savefig('Plots/comp.png', dpi=300)
    # plt.show()

# sim_info = [
#             r'$10^{14} M_{\odot}$',
#             r'$10^{13} M_{\odot}$'    
#                 ]

sim_info = [ r'$10^{13} M_{\odot}$', r'$10^{14} M_{\odot}$']

sim_data([e13_files, e14_files], 1, sim_info)


