import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import glob
from scipy.stats import binned_statistic
from tqdm import tqdm

# colors= ['#32467d', '#4d585a', '#4e6a93', '#718898', '#a09743', '#a3ad93']
import cmasher as cmr
colors = ['#30011E', '#698964', '#00A5CF']

#Paths
# fid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/light_jet/output_M13j45t128/'
# heavy_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/heavy_jet/output_M13j45t128/'
# low_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Gas_fraction_tests/low_gas_jet/output_M13j45t128/'

light_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/light_jet/output_M14j45t128/' 
heavy_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/heavy_jet/output_M14j45t128/'
less_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Gas_fraction_tests/low_gas_jet/output_M14j45t128/'

# paths = [fid_path, heavy_path, low_gas_path]
paths = [light_jet_path, heavy_jet_path, less_gas_path]

def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    return radii

def lobe_length(file):
    with h5py.File(file, 'r') as data:
        h = data['Header'].attrs['HubbleParam']
        gas_mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 * 1/ h   #in gm
        jet_tracer = np.array(data['PartType0']['Jet_Tracer'], dtype=np.float64) 

        coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) * 1 / h  # coords in kpc
        coords = coords - data['Parameters'].attrs['BoxSize']*1/h * 0.5
        axis_mask = (np.abs(coords[:, 1]) < 10) & (np.abs(coords[:, 2]) < 10)
        
        tracer_mask = jet_tracer > 1e-4
        
        jet_mask = axis_mask & tracer_mask
        
        jet_coords = coords[jet_mask]
        jet_radii = np.linalg.norm(jet_coords, axis=1) # will break if bh is not at center, should be fine for now
        jet_mass_tot = gas_mass[jet_mask]

        jet_mass = jet_mass_tot * jet_tracer[jet_mask]

        sorting_logic = np.argsort(jet_radii)
        sorted_radii = jet_radii[sorting_logic]
        sorted_jet_mass = jet_mass[sorting_logic]

        cumulative = np.cumsum(sorted_jet_mass)

        if len(cumulative) == 0:
            jet_head_radius = 0
        else:
            index  = np.argmax(cumulative >= 0.99 * cumulative[-1]) # maybe it makes more sense to compare only to the masked tracers and ignore risen jet
            jet_head_radius = sorted_radii[index]

        return jet_head_radius * 3.085678e+21 # cm


def width_length(file):
    with h5py.File(file, 'r') as data:
        h = data['Header'].attrs['HubbleParam']
        gas_mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 * 1/ h   #in gm
        jet_tracer = np.array(data['PartType0']['Jet_Tracer'], dtype=np.float64) 

        coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) * 1 / h  # coords in kpc
        coords = coords - data['Parameters'].attrs['BoxSize']*1/h * 0.5
        # axis_mask = (np.abs(coords[:, 0]) < 5)
        
        tracer_mask = jet_tracer > 1e-4
        
        # jet_mask = axis_mask & tracer_mask
        
        jet_coords = coords[tracer_mask]

        xs = jet_coords[:, 0] # isolate x coords
        n_bins = 30

        if len(xs) == 0:
            return 0
        else:
            x_bins = np.linspace(min(xs), max(xs), n_bins + 1) # the issue being there is no need for linspace to match the actual bin value.

            # x_bin_centers
            widths = []
            for i in range(n_bins):
                index = np.where((jet_coords[:, 0] > x_bins[i]) & (jet_coords[:, 0] < x_bins[i+1] ) )
                if len(index) == 0:
                    widths.append(0)
                else:
                    ys, zs = jet_coords[index, 1:].T
                    width = np.sqrt(ys*ys + zs*zs)
                    widths.append(np.mean(width))

            max_width = max(widths)

            return max_width * 3.085678e+21 # cm


def check_time(file):
    with h5py.File(file, 'r') as data:
        time_conversion_factor = 3.08568e21 / 100000 * u.s
        h = data['Header'].attrs['HubbleParam']
        time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h
    return time



def radius_width_calc(file):
    
    jet_radius = lobe_length(file) # calculating twice bad code

    jet_width = width_length(file)

    return jet_radius, jet_width

if __name__ == '__main__':
    plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
        })

    fig, ax = plt.subplots(figsize = (6,4))

    labels = ['fiducial jet', 'heavy jet', 'lower gas mass']

    for i, path in enumerate(paths):
        files = sorted(glob.glob(path+"*.hdf5"))
        lengthbywidths = []
        times = []

        for file in tqdm(files[2:]): #Ignoring 0th snap because it doesn't have any jet tracers yet
            # print(file)
            jet_radius, jet_width = radius_width_calc(file)

            if jet_width == 0:
                lengthbywidths.append(0)
            else:
                lengthbywidths.append(jet_radius / jet_width)
            times.append(check_time(file).value)


        ax.plot(times, lengthbywidths, color=colors[i], label = labels[i]) 

    from matplotlib.ticker import ScalarFormatter
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    ax.set_yscale('log')
    ax.set_xlabel('Time (Myr)')
    ax.set_ylabel(r'$R_{\mathrm{jet}} / R_{jet\_width}$') # not in [\mathrm{cm}^{-2}] anymore
    plt.tight_layout()
    plt.legend()
    plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/1e14_3run_lobebywidth.png', dpi = 300, bbox_inches='tight')





