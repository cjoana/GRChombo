"""
Initial Data generator for Curvature (Gaussian) perturbation in Radiation domination   --- by Cristian Joana
"""


"""
SET PARAMETERS
"""
# Perturbation pars KA, LA = Amplitude, width in Gauss
KA= 1.4
LA = 1

# paths 
path = "./"
filename_ext = path +"new_data.3d.hdf5"  # Name of the new file to create
EXTENDED = True  # Extended H5 (with ricci, trA2, HamRel, etc)

# grid pars
N = 256
L = 10
dt_multiplier = 0.01

#fluid pars
omega = 0.333333333333333333
mass = 1  # 1  ALWAYS!  --> : enthalpy = (mass + energy + pressure/density)
rho_mean = 1.0


#intial params for initial D
convergence = 1e-15
D_ini = rho_mean * 0.0001


"""
Loading and starting code
"""
import numpy as np
import os

import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import ndimage



def G_prof(r, A = KA, Lamb = LA ):
    return A*np.exp(-r**2 / (2*Lamb**2))

def K_prof(r, A = KA, Lamb = LA ):     
    return A*np.exp(-r**2 / (2*Lamb**2))

def Curv(r, cutoff = np.inf):
    return integrate.quad(lambda ir:  (1 - (1 - K_prof(ir)*ir**2)**-0.5 )/ir , cutoff, r)


def curv_function(x, y, z, L=1):
    cord = np.array([x, y, z])
    c0 = np.array([L/2, L/2, L/2])
    v = cord - c0
    r = np.sqrt(np.dot(v.T, v))
    
    return Curv(r)[0]

def radius(x,y, z, L=10):
    cord = np.array([x, y, z])
    c0 = np.array([L/2, L/2, L/2])
    v = cord - c0
    r = np.sqrt(np.dot(v.T, v))
    return r


def second_derivative(f, dx): 
    sigma = 1
    mode = 'wrap'
    krn = [-1/12, 4/3, -5/2, 4/3, -1/12]
    nw = 10
    f2 = np.hstack([f[-nw: ], f, f[:nw]])
    ccf2 = ndimage.filters.convolve(f2, krn, mode=mode) [nw:-nw] / dx / dx
    
    return ccf2



import h5py as h5
# import yt
import numpy as np
import os



# Set components
if EXTENDED: 
    filename = filename_ext  # Name of the new file to create
    component_names = [  # The order is important: component_0 ... component_(nth-1)
        "chi",

        "h11",    "h12",    "h13",    "h22", "h23", "h33",

        "K",

        "A11",    "A12",    "A13",    "A22", "A23", "A33",

        "Theta",

        "Gamma1", "Gamma2", "Gamma3",

        "lapse",

        "shift1", "shift2", "shift3",

        "B1",     "B2",     "B3",

        "density",  "energy", "pressure", "enthalpy",

        #"u0", "u1", "u2", "u3",

        "D",  "E", "W",

        "Z1", "Z2", "Z3",

        "V1", "V2","V3",

        "Ham",

        #"Ham_ricci", "Ham_trA2", "Ham_K", "Ham_rho",  
        "ricci_scalar", "trA2", "S", "rho", "HamRel",

        "Mom1",   "Mom2",   "Mom3"
    ]
else: 
    component_names = [  # The order is important: component_0 ... component_(nth-1)
        "chi",

        "h11",    "h12",    "h13",    "h22", "h23", "h33",

        "K",

        "A11",    "A12",    "A13",    "A22", "A23", "A33",

        "Theta",

        "Gamma1", "Gamma2", "Gamma3",

        "lapse",

        "shift1", "shift2", "shift3",

        "B1",     "B2",     "B3",

        "density",  "energy", "pressure", "enthalpy",

        #"u0", "u1", "u2", "u3",

        "D",  "E", "W",

        "Z1", "Z2", "Z3",

        "V1", "V2","V3",

        "Ham",

        "Mom1",   "Mom2",   "Mom3"
    ]

temp_comp = np.zeros((N, N, N))   # template for components: array [Nx, Ny. Nz]
dset = dict()
# Here set the value of the components (default: to zero)

dset['chi'] = temp_comp.copy() + 1.
dset['Ham'] = temp_comp.copy()
dset['h11'] = temp_comp.copy() + 1.
dset['h22'] = temp_comp.copy() + 1.
dset['h33'] = temp_comp.copy() + 1.
dset['lapse'] = temp_comp.copy() + 1.

dset['D'] = temp_comp.copy()
dset['E'] = temp_comp.copy()
dset['density'] = temp_comp.copy()
dset['energy'] = temp_comp.copy()
dset['pressure'] = temp_comp.copy()
dset['enthalpy'] = temp_comp.copy()
dset['Z0'] = temp_comp.copy()
dset['u0'] = temp_comp.copy() + 1.
dset['W'] = temp_comp.copy() + 1.
dset['K'] = temp_comp.copy() #+ k_val

rho_emtensor = temp_comp.copy()

curvature = temp_comp.copy()
psi = temp_comp.copy()
th_drho = temp_comp.copy()

# ## Constructing variables (example for SF)
indices = []
for z in range(N):
    for y in range(N):
        for x in range(N):
            #wvl = 2 * np.pi * 4 / L
            ind = x + y*N + z*N**2 
            
            dd = L/N
            
            r = radius(x * dd , y * dd , z * dd, L=L)

            #curvature[x][y][z] = curv_function(x * dd, y *dd, z *dd, L=L)                       
            curvature[x][y][z] = G_prof(r)                       
            psi[x][y][z] = np.exp( curvature[x][y][z]/2 )
            dset['chi'][x][y][z] = psi[x][y][z]**-4

            th_drho[x][y][z] = (1/2) * (1 - (r/LA)**2 * (1 + curvature[x][y][z]/2)/3 ) * \
                                  curvature[x][y][z] / (LA**2 * psi[x][y][z]**4)            

            indices.append(ind)


dpx = temp_comp.copy()     
dpy = temp_comp.copy()
dpz = temp_comp.copy()
for y in range(N):
        for x in range(N):
            dx = L/N
            dpx[:, x, y] = second_derivative( psi[:, x, y], dx)
            dpy[x, :, y] = second_derivative( psi[x, :, y], dx )
            dpz[x, y, :] = second_derivative( psi[x, y, :], dx )


lap = (psi ** -5) *( dpx + dpy + dpz) 
drho = th_drho

#W2 =  (-(psi ** -5) *( dpx + dpy + dpz) + 2*np.pi*rho_mean ) / (  2*np.pi*(rho_mean +drho ))
#dset['W'] = np.sqrt(W2) 





print("th_rho: mean {}, min {}, max {}".format(np.mean(drho), np.min(drho), np.max(drho)) )

mean_pressure = 0
error = 1e-4
cnt = 0
while np.abs(error) > convergence:
    D_ini -= error/10
    cnt+=1
    
    if error > 100:
        raise Exception("Sorry, the initial values did not converge") 
    
    print("D_ini = ", D_ini, "   error =", error , end= "\r")
    dset['density'] = temp_comp.copy() + D_ini
    dset['energy'] = (drho + rho_mean )/D_ini 
    #dset['energy'] = (drho + (k_val)**2/( 24 * np.pi) )/D_ini

    dset['D'] = dset['density']
    dset['E'] = dset['density'] * dset['energy']
    dset['enthalpy'] =  1 +  dset['energy'] + omega
    dset['pressure'] =  omega * dset['density'] * ( 1 + dset['energy'])
    
    
    dset['rho'] =  dset['density'] * ( 1 + dset['energy'])
    dset['S'] =  3 * dset['pressure']
    
    mean_pressure = np.mean(dset['pressure'])
    error = mean_pressure - rho_mean * omega 
    
            
print("(final) D_ini = ", D_ini, "   cnt=",cnt, end= "\n")


dset["K"] = - np.sqrt(24*np.pi*dset['rho']  + 12 * lap ) 

print("checks")
print("rho mean:",  rho_mean)
print("Hubb.Rad mean:",  (- 3 * np.mean(dset['K']) )**-1  )
print("K mean:",  np.mean(dset['K']))
print("E mean:",  np.mean(dset['E']))
print("D mean:",  np.mean(dset['D']))
print("min energy:",  np.min(dset['energy']))

print("Pressure mean:",  np.mean(dset['pressure']),  "  ", rho_mean/3)

Ham = lap - dset["K"]**2/12 + 2*np.pi*dset['rho'] 

print("Ham mean, min, max  :: ", np.mean(Ham), np.min(Ham), np.max(Ham) )

save_data = True

if not save_data:
    print("!!!\n\nYou have chosen not to save the data")
else:
        
    if not os.path.exists(path):
        os.mkdir(path)
    print(" ! > new mkdir: ", path)

    """
    Mesh and Other Params
    """
    # def base attributes
    base_attrb = dict()
    base_attrb['time'] = 0.0
    base_attrb['iteration'] = 0
    base_attrb['max_level'] = 0
    base_attrb['num_components'] = len(component_names)
    base_attrb['num_levels'] = 1
    base_attrb['regrid_interval_0'] = 1
    base_attrb['steps_since_regrid_0'] = 0
    for comp,  name in enumerate(component_names):
        key = 'component_' + str(comp)
        tt = 'S' + str(len(name))
        base_attrb[key] = np.array(name, dtype=tt)


    # def Chombo_global attributes
    chombogloba_attrb = dict()
    chombogloba_attrb['testReal'] = 0.0
    chombogloba_attrb['SpaceDim'] = 3

    # def level0 attributes
    level_attrb = dict()
    level_attrb['dt'] = float(L)/N * dt_multiplier
    level_attrb['dx'] = float(L)/N
    level_attrb['time'] = 0.0
    level_attrb['is_periodic_0'] = 1
    level_attrb['is_periodic_1'] = 1
    level_attrb['is_periodic_2'] = 1
    level_attrb['ref_ratio']= 2
    level_attrb['tag_buffer_size'] = 3
    prob_dom = (0, 0, 0, N-1, N-1, N-1)
    prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
                        ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
    level_attrb['prob_domain'] = np.array(prob_dom, dtype=prob_dt)
    boxes = np.array([(0, 0, 0, N-1, N-1, N-1)],
          dtype=[('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'), ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])


    """"
    CREATE HDF5
    """

    #TODO: if overwrite:   [...] else: raise()
    if os.path.exists(filename):
        os.remove(filename)

    h5file = h5.File(filename, 'w')  # New hdf5 file I want to create

    # base attributes
    for key in base_attrb.keys():
        h5file.attrs[key] = base_attrb[key]

    # group: Chombo_global
    chg = h5file.create_group('Chombo_global')
    for key in chombogloba_attrb.keys():
        chg.attrs[key] = chombogloba_attrb[key]

    # group: levels
    l0 = h5file.create_group('level_0')
    for key in level_attrb.keys():
        l0.attrs[key] = level_attrb[key]
    sl0 = l0.create_group('data_attributes')
    dadt = np.dtype([('intvecti', '<i4'), ('intvectj', '<i4'), ('intvectk', '<i4')])
    sl0.attrs['ghost'] = np.array((3, 3, 3),  dtype=dadt)
    sl0.attrs['outputGhost'] = np.array( (0, 0, 0),  dtype=dadt)
    sl0.attrs['comps'] = base_attrb['num_components']
    sl0.attrs['objectType'] = np.array('FArrayBox', dtype='S10')

    # level datasets
    dataset = np.zeros((base_attrb['num_components'], N, N, N))
    for i, comp in enumerate(component_names):
        if comp in dset.keys():
            dataset[i] = dset[comp].T
    fdset = []
    for c in range(base_attrb['num_components']):
        fc = dataset[c].T.flatten()
        fdset.extend(fc)
    fdset = np.array(fdset)

    l0.create_dataset("Processors", data=np.array([0]))
    l0.create_dataset("boxes",  data=boxes)
    l0.create_dataset("data:offsets=0",  data=np.array([0, (base_attrb['num_components'])*N**3]))
    l0.create_dataset("data:datatype=0",  data=fdset)

    h5file.close()
    
print("\n\nDone, the job is finished")
            
