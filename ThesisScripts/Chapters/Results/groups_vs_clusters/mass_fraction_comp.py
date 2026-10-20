import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import glob
from scipy.stats import binned_statistic


# colors = ['#AA5245', '#DAA639', '#6C6CA2' ]
plt.rcParams.update({
            'font.size': 8,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'text.usetex': True,
            'font.family': 'serif',
                     })


labels = ["Fiducial Jet", "Heavy Jet", "Lower gas fraction"]

light_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/light_jet/output_M14j45t128/' 
heavy_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/heavy_jet/output_M14j45t128/'
less_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Gas_fraction_tests/low_gas_jet/output_M14j45t128/'

m14_paths = [light_jet_path, heavy_jet_path, less_gas_path]

fid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/light_jet/output_M13j45t128/'
heavy_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/heavy_jet/output_M13j45t128/'
low_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Gas_fraction_tests/low_gas_jet/output_M13j45t128/'

m13_paths = [fid_path, heavy_path, low_gas_path]

sim_colors = ['#AA5245', '#DAA639', '#6C6CA2']


def output_files(path):
    return sorted(glob.glob(path + '*.hdf5'))


def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21  # now in cm
    return radii


def r500_calc(data):
    # calculating r_500 based on 500 times the critical density.
    G = 6.674e-8
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) *1e2 * 1e5 / 3.086e24
    rho_crit = 3 * H**2 / (8 * np.pi * G)

    rho_500 = rho_crit * 500

    # now i need to load masses and volumes (or calculate volume using density) and make spherical shells of the density to find r500
    h = data['Header'].attrs['HubbleParam']
    mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 * 1 / h
    # density = np.array(data['PartType0']['Density'], dtype=np.float64) * 6.769911178294544e-22 * h**2
    # volume = mass / density
    radii = radii_calc(data)

    sorting_logic = np.argsort(radii)
    sorted_radii = radii[sorting_logic]
    sorted_mass = mass[sorting_logic]
    # sorted_volume = volume[sorting_logic]

    n_bins = 100
    radii_bins = np.geomspace(sorted_radii[sorted_radii > 0][0], sorted_radii[-1], n_bins + 1)  # np.linspace(sorted_radii[0], sorted_radii[-1], n_bins+1)
    mass_bins, _, _ = binned_statistic(sorted_radii, sorted_mass, statistic=np.sum, bins=radii_bins)
    vol_bins = 4 / 3 * np.pi * radii_bins[1:]**3
    # vol_bins, _, _ = binned_statistic(sorted_radii, sorted_volume, statistic=np.sum, bins = radii_bins)

    # also using dm mass

    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 * 1 / h  # Now in gm
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) * 1e2 * 1e5 / 3.086e24

    G = 6.674e-8
    r200 = (G * m200 / (100 * H**2))**(1 / 3)
    r_s = r200 / c
    rho_0 = m200 / (4 * np.pi * r_s**3 * (np.log(1 + c) - c / (1 + c)))
    dm_enclosed = 4 * np.pi * rho_0 * r_s**3 * (np.log((r_s + radii_bins[1:]) / r_s) - radii_bins[1:] / (r_s + radii_bins[1:]))
    density_bins = (np.cumsum(mass_bins) + dm_enclosed) / vol_bins
    # density_bins = np.cumsum(mass_bins) / vol_bins

    dens_diff = density_bins - rho_500
    index = np.argmin(np.abs(dens_diff))  # np.where(np.abs(dens_diff) == np.min(abs(dens_diff)))

    return radii_bins[1:][index]


def mass_frac_calc(data, r_500):
    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 * 1 / h  # Now in gm
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) * 1e2 * 1e5 / 3.086e24

    G = 6.674e-8
    r200 = (G * m200 / (100 * H**2))**(1 / 3)
    r_s = r200 / c

    rho_0 = m200 / (4 * np.pi * r_s**3 * (np.log(1 + c) - c / (1 + c)))

    radii = radii_calc(data)
    mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 / h

    mask = radii <= r_500
    radii = radii[mask]
    mass = mass[mask]

    order = np.argsort(radii)
    radii = radii[order]
    mass = mass[order]

    gas_enclosed = np.cumsum(mass)
    dm_enclosed = 4 * np.pi * rho_0 * r_s**3 * (np.log((r_s + radii) / r_s) - radii / (r_s + radii))

    f_gas = gas_enclosed / (gas_enclosed + dm_enclosed)
    r_norm = radii / r_500

    return f_gas, r_norm


def plot_group(ax, paths, colors, lw=1.5, early_idx=1, late_idx=-1):
    early_lw = lw
    for i, path in enumerate(paths):
        files = output_files(path)
        for idx, ls in [(early_idx, '--'), (late_idx, '-')]:
            with h5py.File(files[idx], 'r') as data:
                r_500 = r500_calc(data)
                frac, rad = mass_frac_calc(data, r_500)
            if ls == '--':
                ax.plot(rad, frac, color=colors[i], ls=ls, lw=early_lw)
                early_lw -= 0.3
            else:
                ax.plot(rad, frac, color=colors[i], ls=ls, lw=lw)


def plot():

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, sharex=True,
                              constrained_layout=True)

    plot_group(axes[0], m14_paths, sim_colors, lw=1.8)
    plot_group(axes[1], m13_paths, sim_colors, lw=1.8)

    axes[0].set_title(r'Galaxy Cluster ($10^{14}\,M_\odot$)')
    axes[1].set_title(r'Galaxy Group ($10^{13}\,M_\odot$)')

    from matplotlib.lines import Line2D

    sim_handles = [Line2D([0], [0], color=c, lw=1.8, label=l)
                   for c, l in zip(sim_colors, labels)]
    time_handles = [
        Line2D([0], [0], color='k', ls='--', label='Early'),
        Line2D([0], [0], color='k', ls='-', label='Late'),
    ]

    axes[0].legend(handles=sim_handles, loc='lower right', fontsize=7.5, title='Simulation')
    axes[1].legend(handles=time_handles, loc='lower right', fontsize=8, title='Snapshot')

    for ax in axes:
        ax.set_xlim(3e-2, 1)
        ax.set_ylim(1e-3, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$R \, / \, R_{500}$')
        # ax.grid(True, which='both', alpha=0.15)

    axes[0].set_ylabel(r'$f_{\mathrm{gas}}$')

    fig.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/gcvsgg_mass_fraction_comp.png',
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plot()