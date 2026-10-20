import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import glob
from scipy.stats import binned_statistic

# colors= ['#32467d', '#4d585a', '#4e6a93', '#718898', '#a09743', '#a3ad93']
import cmasher as cmr
cmap = cmr.prinsenvlag
colors = cmap(np.linspace(0,1, 5))

#Paths
fid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/light_jet/output_M13j45t128/'
heavy_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/heavy_jet/output_M13j45t128/'
low_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Gas_fraction_tests/low_gas_jet/output_M13j45t128/'

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10, 'text.usetex': True, 'font.family': 'serif',})

def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    return radii

def enclosed_mass_calculator(data):
    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 *1/h # Now in gm
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) *1e2 * 1e5 / 3.086e24
    G = 6.674e-8 
    r200 = (G * m200 / (100*H**2))**(1/3)
    r_s = r200/c
    
    radii = radii_calc(data)

    rho_0 = m200 /(4 * np.pi * r_s**3 * (np.log(1+c) - c/(1+c))) 
    M_enclosed = 4 * np.pi * rho_0 * r_s**3 * ( np.log( (r_s+radii)/r_s) - radii/(r_s + radii))
    mass = np.array(data['PartType0']['Masses'], dtype=np.float64)  * 1.989e+43  *1/h

    # mask = radii < r200
    # radii = radii[mask]
    # mass = mass[mask]
    # M_enclosed = M_enclosed[mask]
    return M_enclosed, mass,  radii

def GBE_calc(data):
    G = 6.674e-8
    m_enc, mass, radius = enclosed_mass_calculator(data)
    GBE = - G * m_enc * mass / radius 

    return GBE, radius, np.sum(GBE)


def delta_E(files_path):

    init_file_path = files_path + r'snap_%03d.hdf5'%1
    final_file_path = files_path + r'snap_%03d.hdf5'%63

    with h5py.File(init_file_path, 'r') as init_data:
        h = init_data['Header'].attrs['HubbleParam']
        init_gbe, r , _ = GBE_calc(init_data)
        log_r = np.logspace(np.log10(5*3.086e+21), np.log10(r.max()), 100)
        init_gbe_binned, _, _ = binned_statistic( r, init_gbe, statistic='sum', bins=log_r)
        # r = 0.5 * (r_edges[1:] + r_edges[:-1])
        # r_bins = np.linspace(0,r[-1], 100)
        # init_gbe_binned = np.histogram(init_gbe, r_bins)
        E_dot = init_data['Config'].attrs['BH_CONSTANT_POWER_RADIOMODE_IN_ERG_PER_S']
        Time_total = init_data['Config'].attrs['BH_CONSTANT_POWER_RADIOMODE_TIMELIMIT_IN_MYR'] * 1/h * u.Myr.to(u.s) 

    with h5py.File(final_file_path, 'r') as final_data:
        final_gbe, r, _ = GBE_calc(final_data)
        log_r = np.logspace(np.log10(5*3.086e+21), np.log10(r.max()), 100)
        final_gbe_binned, r_edges, _ = binned_statistic(r, final_gbe, statistic='sum', bins=log_r)
        r = 0.5 * (r_edges[1:] + r_edges[:-1])


        # Logic for jet head radius
        gas_mass = np.array(final_data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 * 1/ h   #in gm
        jet_tracer = np.array(final_data['PartType0']['Jet_Tracer'], dtype=np.float64) 
        # time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h
        radii = radii_calc(final_data)
        jet_mass = gas_mass * jet_tracer

        sorting_logic = np.argsort(radii)
        sorted_radii = radii[sorting_logic]
        sorted_jet_mass = jet_mass[sorting_logic]

        cumulative = np.cumsum(sorted_jet_mass)

        index  = np.argmax(cumulative >= 0.99 * np.sum(jet_mass))
        jet_head_radius = sorted_radii[index]*u.cm
    
    
    return np.cumsum((final_gbe_binned)) / np.cumsum((init_gbe_binned)) , (r*u.cm).to(u.kpc) , jet_head_radius.to(u.kpc)
if __name__ == '__main__':
    fid_E, fid_r, fid_jet_r = delta_E(fid_path)
    heavy_E, heavy_r, heavy_jet_r = delta_E(heavy_path)
    low_m_E, low_m_r, low_m_jet_r = delta_E(low_gas_path)

    # plt.rcParams.update({
    #     'font.size': 11,
    #     'axes.labelsize': 11,
    #     'xtick.labelsize': 11,
    #     'ytick.labelsize': 11,
    #     'lines.linewidth': 2,
    #     'font.family': 'serif',
    # })


    # plt.rcParams.update({
    # 'text.usetex': True,
    # 'font.family': 'serif',
    #     })

    fig, ax = plt.subplots(figsize = (6,4))
    ax.plot(fid_r, (fid_E), color=colors[0], label = 'Fiducial jet') 
    ax.axvline(fid_jet_r.value, color = colors[0], linestyle = '--', alpha = 0.8)
    ax.plot(heavy_r, (heavy_E), color=colors[1], label = 'Heavy jet')
    ax.axvline(heavy_jet_r.value, color = colors[1], linestyle = '--', alpha = 0.8)
    ax.plot(low_m_r, low_m_E, color=colors[4], label = 'Lower gas fraction')
    ax.axvline(low_m_jet_r.value, color = colors[4], linestyle = '--', alpha = 0.8 )
    ax.set_xscale('log')
    ax.set_xlim(5, 1000)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r$ [kpc]', fontsize=14)
    # ax.set_ylabel(r'$(E_\mathrm{GB}) / (E_\mathrm{GB,\,0})$', fontsize=14) # / E_\mathrm{jet}
    ax.set_ylabel(r'$\mathrm{GBE / GBE_{\mathrm{int}}}$')
    plt.tight_layout()
    plt.legend(loc = 'lower right', framealpha=1)
    plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/3run_del_div_E.png', dpi = 300, bbox_inches='tight')


