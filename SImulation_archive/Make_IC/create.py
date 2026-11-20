# examples/BH_JET_selfregulated_3d/create.py

import numpy as np
import sys
import os
from subprocess import call
from astropy import units as u

from cluster_iso_nfw import profiles
from cluster_iso_nfw import hydrostatic_halo

""" input """
simulation_directory = str(sys.argv[1])

""" get GFM Tables """

def time_codeunit_convert(time):
    time = time * u.Myr.to(u.s)
    h = 0.6774
    code_unit_length = 3.085678e+21
    code_unit_vel  = 100000.0
    code_unit_time = code_unit_length / code_unit_vel

    return time * h / code_unit_time

# parameters
## NOTE: if e13 halo keep Mtarget at e-6 and e-5 if m14 halo 

GRAVITY = 6.6738e-8  ## G in cgs
Mtarget = 6.5e-5 # 0.0004037618637084961         6.5e-5 for e14 halo and 6.5e-6 for e13 halo
BoxSize = 3977.8838577357856
G = GRAVITY / 3.085678e21 ** 3 * 1.989e+43 * (3.085678e+21 / 100000.0) ** 2
gamma = 5./3.
m_bh = 0.067

## additional param
Max_Time = 1000 # In Myr
snapshot_num = 100
Max_Time_codeunits = time_codeunit_convert(Max_Time)




""" derefinement on outskirts """
derefinement_scale_radius = 200.0 # None to deactivate
max_volume = 250**3
print('max volume: ', max_volume)

# matter profile -- see Config.sh
# 1e14 Msun
# ISOmass, ISOradius = 0.2526429932244965, 66.29806429559643
# NFWmass, NFWradius, NFWconcentration = 6774.0, 662.9806429559643, 2.8360053513355195
# 1e13 Msun
ISOmass, ISOradius = 0.2608149607190282, 30.772835492458245
NFWmass, NFWradius, NFWconcentration = 677.4, 307.728, 2.47124496568521

MatterProfile = profiles.isothermal_nfw.IsothermalNFW(ISOmass, ISOradius, NFWmass, NFWradius, NFWconcentration)

# density profile
# 1e14 Msun
# rho0, r0, beta0 = 0.000370736, 17.6091, 1.89242
# rho1, r1, beta1 = 2.35478e-05, 203.302, 1.42512
# 1e13 Msun
rho0, r0, beta0 = 0.000269587, 16.1015, 1.9328
rho1, r1, beta1 = 9.36794e-06, 118.775, 1.0681

# CHANGED BETA PROFILE for 1e14 Msun for lower gas mass fraction ~3.2 % gas fraction
# rho0, r0, beta0 = 0.000081, 17.6091, 1.89242
# rho1, r1, beta1 = 0.85e-05, 203.302, 1.42512
# CHANGED BETA PROFILE for 1e13 Msun for lower gas mass fraction
# rho0, r0, beta0 = 0.000061, 16.1015, 1.9328
# rho1, r1, beta1 = 1.96794e-06, 118.775, 1.0681


GasProfile = profiles.double_beta.doubleBeta(rho0, r0, beta0, rho1, r1, beta1)

myICs = hydrostatic_halo.HydrostaticHalo(MatterProfile,
                                            GasProfile,
                                            Mtarget,
                                            BoxSize=BoxSize,
                                            G=G,
                                            gamma=gamma,
                                            liveCollisionlessMatter=False,
                                            placeBH=m_bh,
                                            setupMetallicity=True,
                                            setupScalar=True,
                                            plot=False,
                                            derefinement_scale_radius = derefinement_scale_radius,
                                            max_volume=max_volume)

refineInCenter = True
densityAsMass = True
myICs.create(simulation_directory+'/ics.hdf5',
            density_as_mass=densityAsMass,
            refineInCenter=refineInCenter)

""" output times """
outputTimes = np.linspace(0, Max_Time_codeunits, snapshot_num)

ones = np.ones(outputTimes.shape, dtype=np.int32)
data = np.array([outputTimes, ones]).T
np.savetxt(simulation_directory+"/output_list.txt",data, fmt="%g %1.f" )
