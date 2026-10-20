import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from scipy import spatial
from matplotlib.colors import LogNorm
import astropy.units as u
import cmasher as cmr

BoxSize = 3977.88
cmap = cmr.sunburst
time_conversion_factor = 3.08568e21 / 1e5 * u.s  

longjet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/cool_j43_t1000_1/output_longjet/'
selffid_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Self_regulated_runs/Fid_self_regulated/output_self_regulated/'

snaps = [5, 25, 49, 74, 96]

longjet_files = [longjet_path + f'snap_{str(i).zfill(3)}.hdf5' for i in snaps]
selffid_files = [selffid_path + f'snap_{str(i).zfill(3)}.hdf5' for i in snaps]

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
})


def get_slice(ax, VoronoiPos, Field, x_min, x_max, y_min, y_max, z_slice,
              vmin, vmax, Nplot, log=False):
    """Plot a single 2D density slice on ax and return the pcolormesh handle."""
    Edges_x = np.linspace(x_min, x_max, Nplot + 1, endpoint=True, dtype=np.float64)
    Edges_y = np.linspace(y_min, y_max, Nplot + 1, endpoint=True, dtype=np.float64)
    Grid_x = 0.5 * (Edges_x[1:] + Edges_x[:-1])
    Grid_y = 0.5 * (Edges_y[1:] + Edges_y[:-1])

    xx, yy = np.meshgrid(Grid_x, Grid_y)
    Grid2D = np.array([
        xx.reshape(Nplot ** 2),
        yy.reshape(Nplot ** 2),
        np.ones(Nplot ** 2) * z_slice
    ]).T
    _, cells = spatial.KDTree(VoronoiPos[:]).query(Grid2D, k=1)

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

    return pc


def get_snapshot_time_myr(file_path):
    try:
        with h5py.File(file_path, 'r') as data:
            time_raw = data['Header'].attrs['Time']
        time_sec = time_raw * time_conversion_factor /.6774
        return time_sec.to(u.Myr).value
    except (KeyError, OSError):
        return None


def plot_multiple_slices(column_files: list, column_labels: list, property: str,
                          vmin, vmax, center, extent,
                          Nplot=256, plot_directory='/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/',
                          log=False, conversion_factor=1, conversion_factor_length=1,
                          cbar_label='Density [g cm$^{-3}$]', output_name='slices.png'):
    """
    column_files: list of lists, one list of snapshot file paths per column
                  (e.g. [longjet_files, selffid_files]). All columns must
                  have the same number of rows (snapshots).
    column_labels: label for each column (e.g. ['Single Outburst', 'Self Regulated'])
    """
    n_cols = len(column_files)
    n_rows = len(column_files[0])
    assert all(len(col) == n_rows for col in column_files), \
        'All columns must have the same number of snapshots'
    assert len(column_labels) == n_cols, \
        'column_labels must match the number of columns'

    x_min = center[0] - 0.5 * extent[0]
    x_max = center[0] + 0.5 * extent[0]
    y_min = center[1] - 0.5 * extent[1]
    y_max = center[1] + 0.5 * extent[1]
    z_slice = center[2]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols+1, 2 * n_rows), dpi=300)
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)
    plt.subplots_adjust(right=0.88, wspace=0.05, hspace=0.05)
    cax = plt.axes([0.90, 0.12, 0.02, 0.75])

    last_pc = None
    for col, (files, label) in enumerate(zip(column_files, column_labels)):
        for row, file_path in enumerate(files):
            ax = axes[row, col]
            print(f'Reading: {file_path}')

            with h5py.File(file_path, 'r') as data:
                VoronoiPos = (np.array(data['PartType0']['Coordinates'], dtype=np.float64)
                              * conversion_factor_length)
                Field = (np.array(data['PartType0'][property], dtype=np.float64)
                         * conversion_factor)

            last_pc = get_slice(ax, VoronoiPos, Field,
                                 x_min, x_max, y_min, y_max, z_slice,
                                 vmin, vmax, Nplot, log=log)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

            # Column header on the top row only
            if row == 0:
                ax.set_title(label, fontsize=10)

            # Row label (physical time) on the leftmost column only
            if col == 0:
                time_myr = get_snapshot_time_myr(file_path)
                if time_myr is not None:
                    ax.set_ylabel(f'{time_myr:.0f} Myr', fontsize=8)

            # Scale bar in the first panel only
            if row == 0 and col == 0:
                box_size_kpc = extent[0]
                scale_kpc = round(box_size_kpc / 4, -1)
                x0 = x_min + 0.05 * extent[0]
                y0 = y_min + 0.05 * extent[1]
                ax.plot([x0, x0 + scale_kpc], [y0, y0],
                        color='white', lw=2, solid_capstyle='butt')
                ax.text(x0 + 0.5 * scale_kpc, y0 + 0.02 * extent[1],
                        f'{scale_kpc:.0f} kpc',
                        color='white', ha='center', va='bottom', fontsize=8)

    cb = plt.colorbar(last_pc, cax=cax, extend='both')
    cb.set_label(cbar_label, fontsize=9)
    if log:
        cb.ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
        cb.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
        cb.ax.yaxis.set_minor_locator(ticker.NullLocator())

    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    save_path = os.path.join(plot_directory, output_name)
    plt.savefig(save_path, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close(fig)


if __name__ == '__main__':
    unit_length = 1 / .6774
    unit_density = 1.989e+43 / 3.08568e+21 ** 3 * .6774 ** 2

    extent = np.ones(2) * 0.1 * BoxSize * unit_length
    center = np.ones(3) * 0.5 * BoxSize * unit_length

    plot_multiple_slices(
        column_files=[longjet_files, selffid_files],
        column_labels=['Single Outburst', 'Self Regulated'],
        property='Density',
        vmin=1e-30,
        vmax=1e-25,
        center=center,
        extent=extent,
        log=True,
        conversion_factor=unit_density,
        conversion_factor_length=unit_length,
        output_name='fidvslong_density_slices.png',
    )