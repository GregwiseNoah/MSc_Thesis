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

paths = [      '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/cool_j43_t1000_1/output_longjet'        ]

def get_slice(ax, cax, VoronoiPos, Field, x_min, x_max, y_min, y_max, z_slice,
              vmin, vmax, Nplot, log=False, quiver_field=None, quiver_step=16):
    from scipy import spatial
    Edges_x = np.linspace(x_min, x_max, Nplot + 1, endpoint=True, dtype=np.float64)
    Edges_y = np.linspace(y_min, y_max, Nplot + 1, endpoint=True, dtype=np.float64)
    Grid_x = 0.5 * (Edges_x[1:] + Edges_x[:-1])
    Grid_y = 0.5 * (Edges_y[1:] + Edges_y[:-1])

    xx, yy = np.meshgrid(Grid_x, Grid_y)
    Grid2D = np.array([xx.reshape(Nplot**2), yy.reshape(Nplot**2),
                        np.ones(Nplot**2) * z_slice]).T
    tree = spatial.KDTree(VoronoiPos[:])
    dist, cells = tree.query(Grid2D, k=1)

    if vmin == 0 and vmax == 0:
        vmin, vmax = np.percentile(Field[cells], [1, 99])

    if log:
        pc = ax.pcolormesh(Edges_x, Edges_y, Field[cells].reshape((Nplot, Nplot)),
                            rasterized=True, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        pc = ax.pcolormesh(Edges_x, Edges_y, Field[cells].reshape((Nplot, Nplot)),
                            rasterized=True, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(pc, cax=cax)

    if quiver_field is not None:
        # subsample onto a coarser grid so arrows are visible
        qx = xx[::quiver_step, ::quiver_step]
        qy = yy[::quiver_step, ::quiver_step]
        qcells = cells.reshape(Nplot, Nplot)[::quiver_step, ::quiver_step]
        U = quiver_field[qcells.reshape(-1), 0].reshape(qx.shape)
        V = quiver_field[qcells.reshape(-1), 1].reshape(qx.shape)
        mag = np.sqrt(U**2 + V**2)
        mag[mag == 0] = 1  # avoid div by zero
        Un, Vn = U/mag, V/mag
        dx = (x_max - x_min) / Nplot  # size of one grid cell in data units
        arrow_len = dx * quiver_step * 0.8  # arrows slightly smaller than their cell spacing
        ax.quiver(qx, qy, Un, Vn, color='yellow', scale_units='xy', scale=1/arrow_len, pivot='mid')
        
plot_dir = '/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/self_comp_slices' # OTHERWISE './slices/'
def plot_slice(directory, snapshot, property, vmin, vmax, center, extent, Nplot = 256, plot_directory = plot_dir, log = False, conversion_factor = 1, conversion_factor_length = 1):
    os.chdir(directory)

    file_path = directory + r'/snap_%03d.hdf5'%snapshot
    print(file_path)

    with h5py.File(file_path, 'r') as data:
        VoronoiPos = np.array(data['PartType0']['Coordinates'], dtype = np.float64) * conversion_factor_length#* 1/h
        Vel = np.array(data['PartType0'][property], dtype = np.float64) * conversion_factor #* 1/h

        Field = np.linalg.norm(Vel, axis=1)
        h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
        time = data['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr) *1/h

    x_min, x_max = center[0] - 0.5*extent[0], center[0] + 0.5*extent[0]
    y_min, y_max = center[1] - 0.5*extent[1], center[1] + 0.5*extent[1]
    z_slice = center[2]
    
    fig = plt.figure(figsize=np.array([7.5, 6.0]), dpi=300)
    ax = plt.axes([0.1, 0.10, 0.75, 0.85])
    cax = plt.axes([0.85, 0.1, 0.02, 0.85])
    print('Making the plot')
    get_slice(ax, cax, VoronoiPos, Field, x_min, x_max, y_min, y_max, z_slice, vmin, vmax, Nplot, log=log, quiver_field=None, quiver_step=16)
    print('Made plot')
    ax.set_title(f'{time:.0f}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # save_plot_dir = os.path.join(directory, plot_directory)
    # if not os.path.exists(save_plot_dir):
    #     os.mkdir(save_plot_dir)

    # print(
    #     os.path.join(save_plot_dir,
    #                     f'{property}_%03d.png' % snapshot))
    # fig.savefig(os.path.join(save_plot_dir,
    #                             f'{property}_%03d.png' % snapshot),
    #             dpi=300)

    if not os.path.exists(plot_directory):
        os.mkdir(plot_directory)

    print(
        os.path.join(plot_directory,
                        f'{property}_%03d.png' % snapshot))
    fig.savefig(os.path.join(plot_directory,
                                f'{property}_%03d.png' % snapshot),
                dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    unit_length = 1 / .6774
    unit_density = 1.989e+43 / 3.08568e+21**3 * .6774**2
    unit_velocity = 1e5  / .6774
    extent = np.ones(2)* 0.1*BoxSize * unit_length # (50*u.kpc).to(u.cm).value
    center = np.ones(3)*0.5*BoxSize * unit_length

    # Making assumption that all runs have 64 output files

    snapshot_range = np.linspace(0,99, 100, dtype=np.int32) ####### CHANGE HERE FOR SELF REGULATED
    for path in paths:
        for snapshot_number in snapshot_range:
            try:
                plot_slice(path, snapshot_number, 'Velocities', vmin = 0, vmax= 0, center = center, extent = extent, Nplot = 512, plot_directory=f'./vel_slices_{(2*extent[0]*u.kpc):.0f}/', log = False, conversion_factor = unit_velocity, conversion_factor_length = unit_length )
            except FileNotFoundError:
                print(f'Skipping: {path} snapshot {snapshot_number} not found')
            except Exception as e:
                print(f'Error processing {path} snapshot {snapshot_number}: {e}')
    # plot_slice(path_j46t12_8, 62, 'Density', vmin = 1e-30, vmax= 1e-25, center = center, extent = extent, log = True, conversion_factor = unit_density, conversion_factor_length = unit_length )
    # plot_multiple_slices([path_j46t12_8], 32, ['Density', 'Jet_Tracer'], vmins = [1e-30, 1e-8], vmaxs= [1e-25, 1e-2], center = center, extent = extent, logs = [True, True], conversion_factors = [unit_density, 1], conversion_factor_length = unit_length )
