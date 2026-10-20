import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import glob
from tqdm import tqdm
# import cmasher as cmr


# cmap = cmr.prinsenvlag
# colors = cmap(np.linspace(0, 1, 6))

colors = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']


#paths 
self_fid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/Fid_self_regulated/output_self_regulated/'
self_lowgas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/low_gas_jet/output_self/'


self_fid_files = sorted(glob.glob(self_fid_path+'*.hdf5'))
self_lowgas_files = sorted(glob.glob(self_lowgas_path   +'*.hdf5'))
# self_heavyjet_files = sorted(glob.glob(self_heavyjet_path   +'*.hdf5'))


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
        jet_head_radius=0
    else:
        index  = np.argmax(cumulative >= 0.99 * cumulative[-1]) # maybe it makes more sense to compare only to the masked tracers and ignore risen jet
        jet_head_radius = sorted_radii[index]


    time_conversion_factor = 3.08568e21 / 100000 * u.s
    time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h

    return jet_head_radius* 3.085678e+21, time



def jet_mass_check_self(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    gas_mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 * 1/ h   #in gm
    jet_tracer = np.array(data['PartType0']['Jet_Tracer'][:,1], dtype=np.float64) 
    # time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h

    coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) * 1 / h  # coords in kpc
    coords = coords - data['Parameters'].attrs['BoxSize']*1/h * 0.5
    axis_mask = (np.abs(coords[:, 1]) < 50) & (np.abs(coords[:, 2]) < 50)
    
    tracer_mask = jet_tracer > 1e-6
    
    jet_mask = axis_mask & tracer_mask
    
    # print(coords.min(axis=0))
    # print(coords.max(axis=0))
    # print(np.median(coords, axis=0))
    # print("coords shape:", coords.shape)
    # print("jet tracer min/max:", jet_tracer.min(), jet_tracer.max())

    # print("axis particles:", np.sum(axis_mask))
    # print("tracer particles:", np.sum(tracer_mask))
    # print("combined:", np.sum(jet_mask))

    jet_coords = coords[jet_mask]
    jet_radii = np.linalg.norm(jet_coords, axis=1) # will break if bh is not at center, should be fine for now
    jet_mass_tot = gas_mass[jet_mask]
    # print(jet_radii)
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

# Adding a r200 line 
def r200_calc(data):
    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 *1/h # Now in gm
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) *1e2 * 1e5 / 3.086e24
    # R_vir not in config  
    # From page 45, equation for r200/rvir
    G = 6.674e-8 
    r200 = (G * m200 / (100*H**2))**(1/3) * u.cm

    return r200.to(u.kpc)


def jet_mass_files(files):
    jet_head_radii = []
    times = []
    for afile in tqdm(files):
        
        with h5py.File(afile, 'r') as data:
            jet_head_radius, time = jet_mass_check(data)
        jet_head_radii.append(jet_head_radius)
        times.append(time)
    return (np.array(jet_head_radii) * u.cm).to(u.kpc), u.Quantity(times)

def jet_mass_files_self(files):
    jet_head_radii = []
    times = []
    for afile in tqdm(files):
        
        with h5py.File(afile, 'r') as data:
            jet_head_radius, time = jet_mass_check_self(data)
        jet_head_radii.append(jet_head_radius)
        times.append(time)
    return (np.array(jet_head_radii) * u.cm).to(u.kpc), u.Quantity(times)




if __name__ == '__main__':
    time_conversion_factor = 3.08568e21 / 100000 * u.s


    with h5py.File(self_fid_files[0], 'r') as data:
        r200 = r200_calc(data)
    # # print(r200)
    # with h5py.File(fid_files[0], 'r') as data:
    #     r200 = r200_calc(data)
    # print(r200)

    time_conversion_factor = 3.08568e21 / 100000 * u.s

    self_fid_radii, self_fid_times          = jet_mass_files_self(self_fid_files)
    # self_heavy_radii, self_heavyjet_times   = jet_mass_files_self(self_heavyjet_files)
    self_lowg_radii, self_lowg_times         = jet_mass_files_self(self_lowgas_files)


    plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
        })
        
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(self_fid_times, self_fid_radii, color=colors[0], label = 'fiducial')
    # ax.plot(self_heavyjet_times, self_heavy_radii,  color=colors[1], label ='heavy jet')
    ax.plot(self_lowg_times, self_lowg_radii, color=colors[4], label = 'lower gas fraction')

    ax.axhline(y = r200.value, color = '#808080', ls='--', label = r'$R_{200}$')
    ax.set_yscale('log')
    ax.set_ylabel('Radius [kpc]')
    ax.set_xlabel('Time [Myr]')
    # ax.set_ylim(3, 300)
    # ax.set_xlim(3, 300)
    ax.legend(loc = 'lower right')
    plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/self_3jet_jethead.png', dpi=300, bbox_inches='tight')