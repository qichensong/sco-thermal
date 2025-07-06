import numpy as np
import matplotlib.pyplot as plt
from constants import meV2thz

def intQ(file_index,datai,data_ei,q,w,T,Vol,natom):
    fig = plt.figure(51+file_index,figsize=(5,4.4))
    ax = fig.add_subplot(111)
    data = datai
    data_e = data_ei
    data[data<-0.5] = 0
    data_e[data_e<-0.5] = 0
    d1 = np.zeros((data.shape[1],))
    dos = np.zeros((data.shape[1],))
    d1e = np.zeros((data.shape[1],))
    for i in range(data.shape[1]):
        d1[i] = np.trapz(data[:,i],q[:])
        dos[i] = np.trapz(data[1:,i]/q[1:]**2*w[i]*(1-np.exp(-w[i]/(T*25.7/298))),q[1:])
        d1e[i] = np.trapz(data_e[:,i]**2,q)
    d1e = np.sqrt(d1e)
    ax.errorbar(w,d1,d1e,ecolor='grey',lw=2)
    ax.set_yscale('log')
    ax.set_xlim([-25,2.6])
    ax.set_ylim([0.001,1])
    dos[w>-0.2] = 0
    idx = np.where(w<=-0.2)
    energy = np.append(-w[idx],0.0)
    energy = np.flip(energy)
    gdos = np.append(dos[idx],0.0)
    gdos = np.flip(gdos)
    N0 = natom * 3 / Vol/ 1e-30
    # The signal above 39.5 meV becomes weaker as energy increases and can contain non-phonon contributions
    idx_local = np.where(energy<39.5/meV2thz)[0]
    area = np.trapz(gdos[idx_local],energy[idx_local])
    gdos = gdos/area*N0
    scaling = 1/area*N0
    return scaling, energy, gdos, area
    