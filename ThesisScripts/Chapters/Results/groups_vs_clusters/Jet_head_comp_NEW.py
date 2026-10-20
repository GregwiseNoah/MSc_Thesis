import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import glob
from tqdm import tqdm
from matplotlib.lines import Line2D
# import cmasher as cmr


# cmap = cmr.prinsenvlag
# colors = cmap(np.linspace(0, 1, 6))
colors = ['#E84855', '#B56B45', '#2B3A67']

#group paths
g_light_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/light_jet/output_M13j45t128/'
g_heavy_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/heavy_jet/output_M13j45t128/'
g_low_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Gas_fraction_tests/low_gas_jet/output_M13j45t128/'


#cluster paths

c_light_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/light_jet/output_M14j45t128/' 
c_heavy_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/heavy_jet/output_M14j45t128/'
c_less_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Gas_fraction_tests/low_gas_jet/output_M14j45t128/'


g_light_jet_files = sorted(glob.glob(g_light_path+ "snap*"))
g_heavy_jet_files = sorted(glob.glob(g_heavy_path+ "snap*"))
g_less_gas_files = sorted(glob.glob(g_low_gas_path+ "snap*"))

c_light_jet_files = sorted(glob.glob(c_light_jet_path+ "snap*"))
c_heavy_jet_files = sorted(glob.glob(c_heavy_jet_path+ "snap*"))
c_less_gas_files = sorted(glob.glob(c_less_gas_path+ "snap*"))



def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    return radii

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
        jet_head_radius=1
    else:
        index  = np.argmax(cumulative >= 0.99 * cumulative[-1]) # maybe it makes more sense to compare only to the masked tracers and ignore risen jet
        jet_head_radius = sorted_radii[index]


    time_conversion_factor = 3.08568e21 / 100000 * u.s
    time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h

    return jet_head_radius* 3.085678e+21, time


def jet_mass_files(files):
    jet_head_radii = []
    times = []
    for afile in tqdm(files):
        
        with h5py.File(afile, 'r') as data:
            jet_head_radius, time = jet_mass_check(data)
        jet_head_radii.append(jet_head_radius)
        times.append(time)
    return (np.array(jet_head_radii) * u.cm).to(u.kpc), u.Quantity(times)
# def plot():
#     return None #plot here


if __name__ == '__main__':
    time_conversion_factor = 3.08568e21 / 100000 * u.s


    g_fid_radii, g_fid_times = jet_mass_files(g_light_jet_files)
    g_heavy_radii, g_heavy_times = jet_mass_files(g_heavy_jet_files)
    g_lowg_radii, g_lowg_times = jet_mass_files(g_less_gas_files)

    c_fid_radii, c_fid_times = jet_mass_files(c_light_jet_files)
    c_heavy_radii, c_heavy_times = jet_mass_files(c_heavy_jet_files)
    c_lowg_radii, c_lowg_times = jet_mass_files(c_less_gas_files)
    
    plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
        })
        
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(g_fid_times, g_fid_radii, color=colors[0], ls='-', label = 'fiducial')
    ax.plot(c_fid_times, c_fid_radii, color=colors[0], ls='--')

    ax.plot(g_heavy_times, g_heavy_radii,  color=colors[1], ls='-', label ='heavy jet')
    ax.plot(c_heavy_times, c_heavy_radii,  color=colors[1], ls='--')

    ax.plot(g_lowg_times, g_lowg_radii, color=colors[2], ls='-', label = 'lower gas fraction')
    ax.plot(c_lowg_times, c_lowg_radii, color=colors[2], ls='--')

    
    ax.set_yscale('log')
    ax.set_ylabel('Radius [kpc]')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylim(5, 400)
    ax.set_xlim(1, 250)

    color_legend = ax.legend(loc='lower right')
    ax.add_artist(color_legend) 

    linestyle_handles = [
        Line2D([0], [0], color='black', ls='-', label='Galaxy Groups'),
        Line2D([0], [0], color='black', ls='--', label='Galaxy Clusters'),
    ]
    ax.legend(handles=linestyle_handles, loc='upper right')

    plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/gcvsgg_jethead.png', dpi=300, bbox_inches='tight')