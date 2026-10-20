import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import h5py
import glob
import astropy.units as u
import sys
sys.path.append('/home/george/Documents/Code')

# cmap =cmr.ember

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10, 'text.usetex': True, 'font.family': 'serif',})

time_conversion_factor = 3.08568e21 / 100000 * u.s

threshhold = 100 * u.kpc

def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = [i - [boxsize*0.5, boxsize*0.5, boxsize*0.5] for i in Coords]
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    # print(f'Coords: {Coords.min()}, {Coords.max()}; {Coords.shape}; radii: max:{radii.max()}, min:{radii.min()}')
    return radii


def median_binner(data_array, radius, nbins = 100):
    
    # radius_bins = np.logspace(np.log10(radius.min() + 1e-3 ) , np.log10(radius.max()), 100) # Adding a 1e-3 to avoid log10(0)
    radius_bins = np.logspace(np.log10( 5 ) , np.log10(radius.max()), nbins) 

    stat_result = binned_statistic(
        radius,               
        data_array,              
        statistic='median',   
        bins=radius_bins
    )

    binned_array = stat_result.statistic
    bin_edges = stat_result.bin_edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return binned_array, bin_centers

light_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/light_jet/output_M14j45t128/' 
heavy_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/heavy_jet/output_M14j45t128/'
less_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Gas_fraction_tests/low_gas_jet/output_M14j45t128/'


light_jet_files = sorted(glob.glob(light_jet_path+ "snap*"))
heavy_jet_files = sorted(glob.glob(heavy_jet_path+ "snap*"))
less_gas_files = sorted(glob.glob(less_gas_path+ "snap*"))


def enclosed_mass_calculator(data):
    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 *1/h # Now in gm
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) *1e2 * 1e5 / 3.086e24
    # R_vir not in config  
    # From page 45, equation for r200/rvir
    G = 6.674e-8 
    r200 = (G * m200 / (100*H**2))**(1/3)
    # print( G, m200, H)
    r_s = r200/c
    
    radii = radii_calc(data)

    # rho = np.array(data['PartType0']['Density']) * 6.769911178294544e-22
    # rho_0 = rho * ( radii/r_s * (1 + radii/r_s)**2 )
    #Using total mass enclosed (m200) to calculate rho_0
    rho_0 = m200 /(4 * np.pi * r_s**3 * (np.log(1+c) - c/(1+c))) 
    M_enclosed = 4 * np.pi * rho_0 * r_s**3 * ( np.log( (r_s+radii)/r_s) - radii/(r_s + radii))
    # print(r200*u.cm.to(u.kpc))
    mask = radii < r200

    radii = radii[mask]
    mass = np.array(data['PartType0']['Masses'], dtype=np.float64)  * 1.989e+43  *1/h
    mass = mass[mask]
    M_enclosed = M_enclosed[mask]
    return M_enclosed, mass,  radii


# Total/Charateristic GBE calc
def GBE_calc(data):
    G = 6.674e-8
    
    m_enc, mass, radius = enclosed_mass_calculator(data)

    GBE = G * m_enc * mass / radius

    return GBE, radius, np.sum(GBE)



fig, axes = plt.subplots(3, 1, figsize=(4.5, 8.5), sharey=False)
# plt.subplots_adjust(wspace=0.85)

# jet_sets = [
#     (jet_powe44_files, "Jet Power 1e44"),
#     (jet_files, "Jet Power 1e45 "),
#     (jet_powe46_files, "Jet Power 1e46")
# ]

jet_sets = [(light_jet_files, "Fiducial Jet"),(heavy_jet_files, 'Heavy Jet'), (less_gas_files, 'Lower Gas Fraction')]

for ax1, (files, title) in zip(axes, jet_sets):

    ax2 = ax1.twinx()

    print(f'reading {files[0]}')
    with h5py.File(files[0], 'r') as data:
        GBE, radii, _ = GBE_calc(data)
        h = data['Header'].attrs['HubbleParam']

        radius_bins = np.logspace(np.log10(radii.min()), np.log10(radii.max()), 100)
        stat_result = binned_statistic(radii, GBE, bins=radius_bins, statistic='sum')
        binned_gbe = np.cumsum(stat_result.statistic)

        bin_centers = 0.5*(stat_result.bin_edges[1:] + stat_result.bin_edges[:-1])
        bin_centers_kpc = bin_centers * u.cm.to(u.kpc)

        time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) / h

        ax1.plot(bin_centers_kpc, binned_gbe, label=f"GBE(t={time:.0f})")

        E_dot = data['Config'].attrs['BH_CONSTANT_POWER_RADIOMODE_IN_ERG_PER_S']
        Time_total = data['Config'].attrs['BH_CONSTANT_POWER_RADIOMODE_TIMELIMIT_IN_MYR'] * u.Myr.to(u.s) * 1/h
        ax1.axhline(E_dot * Time_total, linestyle='--', color='black', label="Total Jet Energy")
    # initial_gbe = binned_gbe.copy()

    print(f'reading {files[-1]}')
    with h5py.File(files[-1], 'r') as data:
        GBE, radii, _ = GBE_calc(data)
        h = data['Header'].attrs['HubbleParam']

        radius_bins = np.logspace(np.log10(radii.min()), np.log10(radii.max()), 100)
        stat_result = binned_statistic(radii, GBE, bins=radius_bins, statistic='sum')
        binned_gbe = np.cumsum(stat_result.statistic)

        bin_centers = 0.5*(stat_result.bin_edges[1:] + stat_result.bin_edges[:-1])
        bin_centers_kpc = bin_centers * u.cm.to(u.kpc)

        time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) / h

        ax1.plot(bin_centers_kpc, binned_gbe, label=f"GBE(t={time:.0f})")
        # del_e = binned_gbe - initial_gbe

        # print(f"E range:{(del_e/(E_dot*Time_total)).max()}, {(del_e/(E_dot*Time_total)).min()}")
    print(f'looking at jet tracer')
    with h5py.File(files[-1], 'r') as data:
        radii = radii_calc(data)
        h = data['Header'].attrs['HubbleParam']
        tracer = data['PartType0']['Jet_Tracer'][:]
        mass = data['PartType0']['Masses'][:] * 1.989e+43 / h

        radius_bins = np.logspace(np.log10(radii.min()), np.log10(radii.max()), 100)
        stat_result = binned_statistic(radii, tracer * mass, bins=radius_bins, statistic='sum')
        binned_tracermass = stat_result.statistic * u.gram.to(u.M_sun)

        bin_centers = 0.5*(stat_result.bin_edges[1:] + stat_result.bin_edges[:-1])
        bin_centers_kpc = bin_centers * u.cm.to(u.kpc)

        ax2.scatter(bin_centers_kpc, binned_tracermass, s=12, marker='x', color='red',
                    label="Tracer mass")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")


    ax1.set_ylabel("Cumulative GBE (erg)")
    ax2.set_ylabel("Cumulative Tracer mass (M$_\\odot$)", color='red')

    ax1.set_xlim(1, 1e3)
    ax2.set_ylim(1e-4, 1e9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title(title)

axes[2].set_xlabel("Radius (kpc)")

plt.tight_layout()
plt.savefig("/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/1e14_3run_GBE_tracer.png")
plt.show()