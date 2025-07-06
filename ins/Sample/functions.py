import numpy as np
import scipy
from constants import meV, kB
import warnings
from scipy.optimize import OptimizeWarning
# Suppress only the OptimizeWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)
def gaussian( x, b, sigma ):
    return b *np.exp(-(x)**2/2/sigma**2)
def get2theta(y,lambda_neutron):
    return np.arcsin(y*lambda_neutron/4/np.pi)*2/np.pi*180
def readtxtfile(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    nl = len(lines)
    f.close()
    elwidth = 0.7 # energy window for elastic scattering peak, in meV
    nx = int(lines[1].split()[3])
    ny = int(lines[1].split()[4])
    for i in range(nl):
        if lines[i].startswith('SIZE(workspace)'):
            i1 = i+1
        if lines[i].startswith(' [ERRORS]'):
            n2 = len(lines[i-1].split())
            i2 = i-1
        if lines[i].startswith(' [XAXIS]'):
            ix = i+2
        if lines[i].startswith(' [YAXIS]'):
            iy = i+2
        if lines[i].startswith(' [ZAXIS]'):
            iz = i+2
        if lines[i].startswith(' [XTITLE]'):
            ed = i
            break
    data = np.zeros(((i2-i1)*6+n2,))
    data_e = np.zeros(((i2-i1)*6+n2,))
    x = np.zeros((nx,))
    y = np.zeros((ny,))
    for i in range(i1,i2):
        data[(i-i1)*6:(i-i1)*6+6] = np.array(lines[i].split(),dtype=float) 
        data_e[(i-i1)*6:(i-i1)*6+6] = np.array(lines[i-i1+i2+2].split(),dtype=float) 
    data[(i2-i1)*6:(i2-i1)*6+n2] = np.array(lines[i2].split(),dtype=float)
    data_e[(i2-i1)*6:(i2-i1)*6+n2] = np.array(lines[i2-i1+i2+2].split(),dtype=float)
    data = data.reshape((ny,nx))
    data_sub = np.zeros(data.shape)
    data_e = data_e.reshape((ny,nx))
    # The two lines below are customized as we only consider the energy gain data.
    data[data<-0.5] = 0 
    data_e[data_e<-0.5] = 0
    ###############################################

    # energy grid
    for i in range(ix,iy-3):
        x[(i-ix)*6:(i-ix)*6+6] = np.array(lines[i].split(),dtype=float)
    x[(iy-3-ix)*6:(iy-3-ix)*6+np.mod(nx,6)] = np.array(lines[iy-3].split(),dtype=float)
    for i in range(nx):
        if abs(x[i])<1e-4:
            i0 = i
            break
    for i in range(nx):
        if x[i]>=-elwidth:
            iel1 = i
            break
    iel2 = nx-1
    for i in range(iel1,nx):
        if x[i]>elwidth:
            iel2 = i
            break
    # Q grid
    for i in range(iy,iz-3):
        y[(i-iy)*6:(i-iy)*6+6] = np.array(lines[i].split(),dtype=float)
    y[(iz-3-iy)*6:(iz-3-iy)*6+np.mod(ny,6)] = np.array(lines[iz-3].split(),dtype=float)
    xv, yv = np.meshgrid(y, x)
    nz = int(lines[iz-1].split()[3])
    z = np.zeros((nz,))
    for i in range(iz,ed-1):
        z[(i-iz)*6:(i-iz)*6+6] = np.array(lines[i].split(),dtype=float)
    z[(ed-1-iz)*6:(ed-1-iz)*6+np.mod(nz,6)] = np.array(lines[ed-1].split(),dtype=float)
    # elastic scattering
    elastic = np.zeros((ny,2))
    for j in range(ny):
        popt, pcov = scipy.optimize.curve_fit(gaussian,x[iel1:iel2],data[j,iel1:iel2], p0=[data[j,i0],0.3])
        if np.sqrt(pcov[1,1])/popt[1]<0.1 and np.sqrt(pcov[0,0])/popt[0]<0.1:
            elastic[j,:] = popt[:] 
            data_sub[j,:] = data[j,:]-gaussian(x,*popt)
    return data, data_e, elastic, x, y
# calculate the spectral specific heat 
def getCE(T,energy,dos):
    E = energy*meV
    intg = np.zeros(E.shape)
    intg[1:] = E[1:]**2/kB/T**2*np.exp(E[1:]/kB/T)/(np.exp(E[1:]/kB/T)-1)**2*dos[1:] 
    intg[0] = 0
    return intg