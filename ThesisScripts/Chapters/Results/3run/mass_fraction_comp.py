import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import glob
from scipy.stats import binned_statistic
from tqdm import tqdm

colors = ['#AA5245', '#DAA639', '#6C6CA2' ]
plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'text.usetex': True,
            'font.family': 'serif',
                     })

#### WHY 14 halo HERE ?????
# light_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/light_jet/output_M14j45t128/' 
# heavy_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Jet_density_tests/heavy_jet/output_M14j45t128/'
# less_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e14/Gas_fraction_tests/low_gas_jet/output_M14j45t128/'

light_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/light_jet/output_M13j45t128/'
heavy_jet_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Jet_density_tests/heavy_jet/output_M13j45t128/'
less_gas_path = '/lustre/ageorge/Thesis/Simulations/mhalo_1e13/Gas_fraction_tests/low_gas_jet/output_M13j45t128/'


def output_files(path):
    return sorted(glob.glob(path+'*.hdf5'))


#Manually defining the different gg fractions:  
rxc1454_x = [0.01010647822903262, 0.011467600400462607, 0.014127370379020994, 0.018678716683263884, 0.025140881681662803, 0.03045945875192514, 0.036988369410897705, 0.047652530050752484, 0.056524172139385785, 0.0678121841552437, 0.0860363948298595, 0.10217034395589868, 0.1267466211828269, 0.16048509401003477, 0.2019520211001129, 0.24556882347995565, 0.3075895397968589, 0.38175929612041437, 0.4541724825312767, 0.5344206060872494, 0.6114231849875817, 0.6870332335709457, 0.7545824694355121, 0.822325358189498, 0.902617069367859, 0.9792654745594813]
rxc1454_y = [0.002351255473441277, 0.002307868994709766, 0.002227658697895209, 0.002104061922349141, 0.001914109355801939, 0.0019848402539020486, 0.001913015579027195, 0.0019647876796984095, 0.0020865614939532375, 0.0023592765031227327, 0.0026465752026221478, 0.002881372616933852, 0.003660141680551555, 0.00431895989120567, 0.0056358671279974045, 0.007078558693884694, 0.009385698504077965, 0.012124150955778606, 0.015055837304350679, 0.017967471078719113, 0.02082988489822407, 0.02339989572661414, 0.026130327148633326, 0.028711275744770837, 0.03183072710634063, 0.0346727237593837]

rxc0748_x = [0.010275208013211337, 0.011063898383482457, 0.012624897968483421, 0.014622188943235767, 0.01718572457718307, 0.020603498566737943, 0.025171332165160123, 0.030777132362636496, 0.03703797272639582, 0.04529502638064107, 0.05528281917449923, 0.06584962910581889, 0.07677886632034915, 0.09006551570139092, 0.11114554429215981, 0.12959079697706205, 0.14575066131367642, 0.16605351713406147, 0.19125680319265217, 0.22230396306954447, 0.26070998155007963, 0.3033418354914754, 0.35234530040878964, 0.4193120682126486, 0.4972479017206509, 0.5714085104352047, 0.6411087207807777, 0.7118658800781924, 0.7736750472793174, 0.8753079301897692]
rxc0748_y = [0.0020424458307052307, 0.002132500118492484, 0.0023753185624856444, 0.0023363071908531095, 0.0024912589006085045, 0.0026797530981227144, 0.002938613601478786, 0.00325507968163804, 0.0036546727966778355, 0.004087443807218197, 0.004808242701774471, 0.005568782152479774, 0.006301612591558231, 0.007246635724937017, 0.008900790800608141, 0.010064933881193967, 0.011470072444481715, 0.012755260154805874, 0.014421082182740932, 0.01625571042624481, 0.0182963332956588, 0.020542586198724656, 0.023070668917416208, 0.026265955468701578, 0.029515930858724147, 0.03316185344120403, 0.03647052818480453, 0.03946820573211948, 0.042464424910401456, 0.04715599808953657]

a1139_x = [0.010196002179767396, 0.012055426148388315, 0.014323957821713265, 0.017624623926841082, 0.02256996722828308, 0.027298600694805386, 0.03493170357849986, 0.04312866994414621, 0.05302781316255435, 0.06425052587450915, 0.08198051960870238, 0.1009425213203824, 0.12288527261684565, 0.1564908494140826, 0.20045593773385875, 0.25224709284118, 0.30110766176558873, 0.3531179086138343, 0.4299029901946957, 0.4991935397770235, 0.5790510451528836, 0.6493371837757838, 0.7563896679837424, 0.8874370007943133, 0.9997118365188136]
a1139_y = [0.0010139310701876556, 0.001003357894698464, 0.0011061729115243966, 0.0012396136780431605, 0.0014587336252502015, 0.001537850145290015, 0.0017285318963537128, 0.0017157711673150332, 0.0022473466798406004, 0.002602824131632389, 0.003456699200449178, 0.0041260905865924844, 0.005047415223185151, 0.006812770937621911, 0.008641201112735572, 0.011056260231370248, 0.013304336095727344, 0.015550953591051448, 0.018833377692058086, 0.021578028212148943, 0.024545444602029322, 0.027112538692353406, 0.030650906558650135, 0.034657410884537276, 0.03810062017143128]

a1213_x = [0.031763441707670224, 0.03869489219794125, 0.04921650803655676, 0.060398842103628336, 0.07351448922983773, 0.08625114884363451, 0.10957749823488565, 0.13450915555081527, 0.16365893478268595, 0.1992291548228812, 0.2378509300413853, 0.2981586358846828, 0.351184495820736, 0.40604003237510616, 0.4921012545060435, 0.5808980684617125, 0.6500299793394948, 0.736109544250906, 0.8144358558104274, 0.9028121945255667, 0.9982722642072924]
a1213_y = [-0.00007619978197382957, 0.00036860277308871625, 0.0008925218481910755, 0.0013056048767860465, 0.0016417589388906916, 0.0019571312422752016, 0.0025218846503013356, 0.003292997276495831, 0.004209217621473026, 0.005369350187218125, 0.007157675213924508, 0.00939371953375942, 0.011684817284589779, 0.01376554530241105, 0.017936480736768037, 0.021776730992894098, 0.024560757476875736, 0.028612471242785632, 0.032053492976130145, 0.03562722629147693, 0.03979597417228443]

def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    return radii

def r500_calc(data):
    # calculating r_500 based on 500 times the critical density. 
    G = 6.674e-8
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) *1e2 * 1e5 / 3.086e24
    rho_crit = 3 * H**2 / (8 * np.pi * G)

    rho_500 = rho_crit * 500

    #now i need to load masses and volumes (or calculate volume using density) and make spherical shells of the density to find r500
    h = data['Header'].attrs['HubbleParam']
    mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 * 1/ h
    # density = np.array(data['PartType0']['Density'], dtype=np.float64) * 6.769911178294544e-22 * h**2
    # volume = mass / density
    radii = radii_calc(data)

    sorting_logic = np.argsort(radii)
    sorted_radii = radii[sorting_logic]
    sorted_mass = mass[sorting_logic]
    # sorted_volume = volume[sorting_logic]

    n_bins = 100
    radii_bins =  np.geomspace(sorted_radii[sorted_radii > 0][0], sorted_radii[-1], n_bins+1) #np.linspace(sorted_radii[0], sorted_radii[-1], n_bins+1)
    mass_bins, _, _ = binned_statistic(sorted_radii, sorted_mass, statistic=np.sum, bins = radii_bins)
    vol_bins = 4/3 * np.pi * radii_bins[1:]**3
    # vol_bins, _, _ = binned_statistic(sorted_radii, sorted_volume, statistic=np.sum, bins = radii_bins)

    #also using dm mass

    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 *1/h # Now in gm
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) *1e2 * 1e5 / 3.086e24

    G = 6.674e-8 
    r200 = (G * m200 / (100*H**2))**(1/3)
    r_s = r200/c
    rho_0 = m200 /(4 * np.pi * r_s**3 * (np.log(1+c) - c/(1+c))) 
    dm_enclosed = 4 * np.pi * rho_0 * r_s**3 * (np.log((r_s + radii_bins[1:])/r_s) - radii_bins[1:]/(r_s + radii_bins[1:]))
    density_bins = (np.cumsum(mass_bins) + dm_enclosed) / vol_bins
    # density_bins = np.cumsum(mass_bins) / vol_bins 

    dens_diff = density_bins - rho_500
    index = np.argmin(np.abs(dens_diff)) #np.where(np.abs(dens_diff) == np.min(abs(dens_diff)))

    return radii_bins[1:][index]


def mass_frac_calc(data, r_500, n_bins=30):
    c = np.array(data['Config'].attrs['NFW_C'], dtype=np.float64)
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    m200 = np.array(data['Config'].attrs['NFW_M200'], dtype=np.float64) * 1.989e+43 / h
    H = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64) * 1e2 * 1e5 / 3.086e24

    G = 6.674e-8
    r200 = (G * m200 / (100 * H**2))**(1/3)
    r_s = r200 / c
    rho_0 = m200 / (4 * np.pi * r_s**3 * (np.log(1+c) - c/(1+c)))

    radii = radii_calc(data)
    mass = np.array(data['PartType0']['Masses'], dtype=np.float64) * 1.989e+43 / h

    mask = (radii <= r_500) & (radii > 0)
    radii, mass = radii[mask], mass[mask]

    bin_edges = np.logspace(np.log10(radii.min()), np.log10(r_500), n_bins + 1)
    mass_per_bin, _, _ = binned_statistic(radii, mass, statistic='sum', bins=bin_edges)
    gas_enclosed = np.cumsum(mass_per_bin)

    r_eval = bin_edges[1:]
    dm_enclosed = 4 * np.pi * rho_0 * r_s**3 * (
        np.log((r_s + r_eval) / r_s) - r_eval / (r_s + r_eval)
    )

    f_gas = gas_enclosed / (gas_enclosed + dm_enclosed)
    r_norm = r_eval / r_500

    return f_gas, r_norm

# def plot():

#     fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)


#     light_jet_files = output_files(light_jet_path)

#     #plot just for fid right now
#     with h5py.File(light_jet_files[-1], 'r') as data:
#         r_500 = r500_calc(data)
#         fid_f, fid_r = mass_frac_calc(data, r_500)

#     heavy_jet_files = output_files(heavy_jet_path)
#     with h5py.File(heavy_jet_files[-1], 'r') as data:
#         r_500 = r500_calc(data)
#         heavy_f, heavy_r = mass_frac_calc(data, r_500)

#     less_gas_files = output_files(less_gas_path)
#     with h5py.File(less_gas_files[-1], 'r') as data:
#         r_500 = r500_calc(data)
#         lowg_f, lowg_r = mass_frac_calc(data, r_500)
#     with h5py.File(less_gas_files[2], 'r') as data:
#         r_500 = r500_calc(data)
#         lowg_f_i, lowg_r_i = mass_frac_calc(data, r_500)

#     ax.plot(fid_r, fid_f, color=colors[0], label = 'Fiducial Jet')
#     ax.plot(heavy_r, heavy_f, color=colors[1], label = 'Heavy Jet')
#     ax.plot(lowg_r, lowg_f, color=colors[2], label = 'Lower Gas Fraction - Final')
#     ax.plot(lowg_r_i, lowg_f_i, color=colors[2], ls='--', label = 'Lower Gas Fraction - Initial')

#     #lines from the paper
#     ax.plot(rxc0748_x, rxc0748_y, alpha=0.6, label = 'RXC0748')
#     ax.plot(rxc1454_x, rxc1454_y, alpha=0.6, label = 'RXC1454')
#     ax.plot(a1139_x, a1139_y, alpha=0.6, label = 'A1139')
#     ax.plot(a1213_x, a1213_y, alpha=0.6, label = 'A1213')

#     # ax.set_yticks(np.linspace(0, 0.7, 8))
#     ax.set_xlim(1e-2, 1)
#     # ax.set_ylim(0, 0.7)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlabel(r'$R \, / \, R_{500}$')
#     ax.set_ylabel(r'$f_{\mathrm{gas}}$')
#     plt.legend(loc='upper left')
#     plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/3run_mass_fraction_comp.png', dpi = 300, bbox_inches='tight')

def load_run(path, initial_idx=2):
    """Return (r, f_gas) for the initial and final snapshot of a run."""
    files = output_files(path)
    with h5py.File(files[initial_idx], 'r') as data:
        r500_i = r500_calc(data)
        f_i, r_i = mass_frac_calc(data, r500_i)
    with h5py.File(files[-1], 'r') as data:
        r500_f = r500_calc(data)
        f_f, r_f = mass_frac_calc(data, r500_f)
    return (r_i, f_i), (r_f, f_f)


def plot():
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # NOTE: assuming snapshot index 0 = "initial" for every run. Your earlier
    # low-gas comparison used index 2 instead - change INITIAL_IDX (or pass a
    # per-run value into load_run) if 0 isn't what you mean by "initial".
    INITIAL_IDX = 0

    runs = [
        ('Fiducial Jet',       light_jet_path, colors[0]),
        ('Heavy Jet',          heavy_jet_path, colors[1]),
        ('Lower Gas Fraction', less_gas_path,  colors[2]),
    ]

    sim_handles = []
    for name, path, color in runs:
        (r_i, f_i), (r_f, f_f) = load_run(path, INITIAL_IDX)
        l_final, = ax.plot(r_f, f_f, color=color, ls='-', lw=2.3,
                            zorder=3, label=f'{name} — Final')
        l_init, = ax.plot(r_i, f_i, color=color, ls='--', lw=1.8, alpha=1,
                           zorder=3, label=f'{name} — Initial')
        sim_handles += [l_final, l_init]

    # observational comparison curves: one muted grey family, distinguished
    # by dash pattern only, drawn behind the sims (zorder=1)
    obs_data = [
        ('RXC1454', rxc1454_x, rxc1454_y, '-'),
        ('RXC0748', rxc0748_x, rxc0748_y, '--'),
        ('A1139',   a1139_x,   a1139_y,   '-.'),
        ('A1213',   a1213_x,   a1213_y,   ':'),
    ]
    obs_handles = []
    for name, x, y, ls in obs_data:
        l, = ax.plot(x, y, color='0.45', ls=ls, lw=1.5, alpha=0.85,
                     zorder=1, label=name)
        obs_handles.append(l)

    # ax.set_yticks(np.linspace(0, 0.14, 8))
    ax.set_xlim(1e-2, 1)
    ax.set_ylim(0.001, .10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$R \, / \, R_{500}$')
    ax.set_ylabel(r'$f_{\mathrm{gas}}$')
    # ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.35, zorder=0)

    # two legends, placed clear of the axes so they never sit on top of data
    leg1 = ax.legend(handles=sim_handles, loc='upper left',
                      title='Simulations',
                      fontsize=9, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=obs_handles, loc='upper right',
              title='Observations',
              fontsize=9, title_fontsize=9)

    plt.savefig('/lustre/ageorge/Thesis/Analysis/ThesisScripts/Plots/3run_mass_fraction_comp.png',
                dpi=300, bbox_inches='tight')


if __name__=='__main__':
    plot()