
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import binned_statistic
import h5py
import astropy.units as u

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'text.usetex': True,
    'font.family': 'serif',
})

KPC_CM = u.kpc.to(u.cm)
DENS_UNIT = 6.769911178294544e-22  # 1e10 Msun / kpc^3 in g / cm^3


def radii_calc(data):
    h = data['Header'].attrs['HubbleParam']
    coords = np.asarray(data['PartType0']['Coordinates'], dtype=np.float64) / h
    boxsize = data['Parameters'].attrs['BoxSize'] / h
    return np.linalg.norm(coords - 0.5 * boxsize, axis=1) * KPC_CM


def median_binner(values, radius, r_min_kpc=10, nbins=30):
    bins = np.logspace(np.log10(r_min_kpc * KPC_CM), np.log10(radius.max()), nbins)
    stat = binned_statistic(radius, values, statistic='median', bins=bins)
    centers = 0.5 * (stat.bin_edges[1:] + stat.bin_edges[:-1])
    return stat.statistic, centers


def load_ic_profile(path, r_max_kpc=1000):
    with h5py.File(path, 'r') as data:
        radii = radii_calc(data)
        h = data['Header'].attrs['HubbleParam']
        density = np.asarray(data['PartType0']['Density']) * DENS_UNIT * h**2

    mask = radii < r_max_kpc * KPC_CM
    rho, r = median_binner(density[mask], radii[mask])
    return r / KPC_CM, rho


halos = [
    dict(logM=13, color='#F06543',
         tng='/lustre/ageorge/Thesis/IllustriTNG/TNG100-1/m13/M13_gas_density_profile_TNG100.txt',
         ic='/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/light_jet/output_M13j45t128/snap_000.hdf5'),
    dict(logM=14, color='#1B3B6F',
         tng='/lustre/ageorge/Thesis/IllustriTNG/M14_gas_density_profile_TNG100.txt',
         ic='/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/light_jet/output_M14j45t128/snap_000.hdf5'),
        ]

fig, ax = plt.subplots(figsize=(6, 4))

for halo in halos:
    c = halo['color']

    r, med, p10, p90 = np.loadtxt(halo['tng'], usecols=(0, 1, 2, 3), unpack=True)
    ax.fill_between(r, p10, p90, color=c, alpha=0.25, lw=0)
    ax.plot(r, med, color=c, ls='-', lw=1.5)

    r_ic, rho_ic = load_ic_profile(halo['ic'])
    ax.plot(r_ic, rho_ic, color=c, ls='--', lw=1.5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(10, 1000)
ax.set_xlabel(r'$r$ [kpc]')
ax.set_ylabel(r'$\rho_{\mathrm{gas}}$ [$\mathrm{g\,cm^{-3}}$]')

mass_handles = [
    Line2D([], [], color=h['color'], lw=2,
           label=rf"$M_{{\mathrm{{halo}}}} = 10^{{{h['logM']}}}\,\mathrm{{M_\odot}}$")
    for h in halos]

mass_legend = ax.legend(handles=mass_handles, loc='upper right')
ax.add_artist(mass_legend)

style_handles = [
    (Patch(facecolor='gray', alpha=0.25, lw=0), Line2D([], [], color='gray', ls='-')),
    Line2D([], [], color='gray', ls='--')]

style_labels = [
    r'TNG100-1 median (10--90\% range)',
    r'Initial conditions']

ax.legend(style_handles, style_labels, loc='lower left')

fig.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/TNG_vs_rwjet_dens.png', dpi=300, bbox_inches='tight')