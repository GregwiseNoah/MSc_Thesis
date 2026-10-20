import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
import astropy.units as u
import cmasher as cmr

BoxSize = 3977.88
cmap = cmr.sunburst
time_conversion_factor = 3.08568e21 / 100000 * u.s

paths = [
    '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/light_jet/output_M14j45t128/snap_032.hdf5',
    '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/light_jet/output_M13j45t128/snap_032.hdf5'
]



def get_slice(ax, cax, VoronoiPos, Field, x_min, x_max, y_min, y_max, z_slice,
              vmin, vmax, Nplot, log=False):
    from scipy import spatial

    Edges_x = np.linspace(x_min, x_max, Nplot + 1, endpoint=True, dtype=np.float64)
    Edges_y = np.linspace(y_min, y_max, Nplot + 1, endpoint=True, dtype=np.float64)
    Grid_x = 0.5 * (Edges_x[1:] + Edges_x[:-1])
    Grid_y = 0.5 * (Edges_y[1:] + Edges_y[:-1])

    xx, yy = np.meshgrid(Grid_x, Grid_y)
    Grid2D = np.array([
        xx.reshape(Nplot**2),
        yy.reshape(Nplot**2),
        np.ones(Nplot**2) * z_slice
    ]).T
    dist, cells = spatial.KDTree(VoronoiPos[:]).query(Grid2D, k=1)

    if log:
        pc = ax.pcolormesh(
            Edges_x, Edges_y,
            Field[cells].reshape((Nplot, Nplot)),
            rasterized=True,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
    else:
        pc = ax.pcolormesh(
            Edges_x, Edges_y,
            Field[cells].reshape((Nplot, Nplot)),
            rasterized=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    if cax is not None:
        cb = plt.colorbar(pc, cax=cax, extend='both')
        cb.set_label('Density [g cm$^{-3}$]', fontsize=9)
        if log:
            import matplotlib.ticker as ticker
            cb.ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
            cb.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
            cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
        return cb


def plot_multiple_slices(directories: list, property: str, vmin, vmax, center, extent,
                         Nplot=256, plot_directory='/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/', log=False,
                         conversion_factor=1, conversion_factor_length=1,
                         output_name='slices.png'):

    labels = ['Galaxy Cluster', 'Galaxy Group']

    plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
        })
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=300)
    plt.subplots_adjust(right=0.88, wspace=0.05)
    # Single shared colorbar on the right
    cax = plt.axes([0.90, 0.1, 0.02, 0.8])

    x_min = center[0] - 0.5 * extent[0]
    x_max = center[0] + 0.5 * extent[0]
    y_min = center[1] - 0.5 * extent[1]
    y_max = center[1] + 0.5 * extent[1]
    z_slice = center[2]

    for i, (file_path, ax) in enumerate(zip(directories, axes)):
        print(f'Reading: {file_path}')

        with h5py.File(file_path, 'r') as data:
            VoronoiPos = (np.array(data['PartType0']['Coordinates'], dtype=np.float64)
                          * conversion_factor_length)
            Field = (np.array(data['PartType0'][property], dtype=np.float64)
                     * conversion_factor)

        # Only draw the colorbar on the last panel
        cax_i = cax if i == len(directories) - 1 else None

        get_slice(ax, cax_i, VoronoiPos, Field,
                  x_min, x_max, y_min, y_max, z_slice,
                  vmin, vmax, Nplot, log=log)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(labels[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Scale bar in the first panel only
        if i == 0:
            box_size_kpc = extent[0]                   # full width of the plotted region in kpc
            scale_kpc    = round(box_size_kpc / 4, -1) # pick a round number ~quarter of the box
            scale_frac   = scale_kpc / box_size_kpc    # fraction of the axis width

            # position: bottom-left corner, 5 % padding from edges
            x0 = x_min + 0.05 * extent[0]
            y0 = y_min + 0.05 * extent[1]
            ax.plot([x0, x0 + scale_kpc], [y0, y0],
                    color='white', lw=2, solid_capstyle='butt')
            ax.text(x0 + 0.5 * scale_kpc, y0 + 0.02 * extent[1],
                    f'{scale_kpc:.0f} kpc',
                    color='white', ha='center', va='bottom', fontsize=8)

    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    save_path = os.path.join(plot_directory, output_name)
    plt.savefig(save_path, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close(fig)


if __name__ == '__main__':
    unit_length = 1 / .6774
    unit_density = 1.989e+43 / 3.08568e+21**3 * .6774**2

    extent = np.ones(2) * 0.1 * BoxSize * unit_length
    center = np.ones(3) * 0.5 * BoxSize * unit_length

    plot_multiple_slices(
        paths,
        property='Density',
        vmin=1e-30,
        vmax=1e-25,
        center=center,
        extent=extent,
        log=True,
        conversion_factor=unit_density,
        conversion_factor_length=unit_length,
        output_name='gcvsgg_density_slice_self_3jet.png',
    )