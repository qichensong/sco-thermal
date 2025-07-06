import matplotlib.pyplot as plt
from constants import meV2thz
from functions import getCE
import matplotlib as mpl

mpl.rcParams['figure.subplot.left'] = 0.18
mpl.rcParams['figure.subplot.right'] = 0.955


def plot_spectral_C(all_energy, all_gdos, all_temperature):
    fig = plt.figure(22,figsize=(5,4.4))
    ax = fig.add_subplot(111)
    for i in range(len(all_temperature)):
        plt.plot(all_energy[i]*meV2thz,getCE(all_temperature[i],all_energy[i],all_gdos[i])/meV2thz/1e5,label=f'{all_temperature[i]} K')
    plt.xlim([5e-2,4e1])
    plt.xlabel(r'Energy $\omega$ (THz)')
    plt.xscale('log')
    plt.ylabel(r'Spectral $C_p(\omega)$ [$\mathregular{\times 10^{5}}$J/(m$\mathregular{^{3}}$$\cdot$K$\cdot$THz)]')
    plt.legend(frameon=False,labelspacing=0.3,handletextpad=0.2)
