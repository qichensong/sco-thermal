import matplotlib.pyplot as plt
from constants import meV2thz
from functions import getCE
import matplotlib as mpl

mpl.rcParams['figure.subplot.left'] = 0.18
mpl.rcParams['figure.subplot.right'] = 0.955


def plot_scatt_phase_space(xs,sp1,sp2,temperatures):
    fig = plt.figure(23,figsize=(5,4.4))
    ax = fig.add_subplot(111)
    for i in range(len(temperatures)):
        ax.plot(xs[i],(sp1[i]+sp2[i]/2)/1000,label=f'{temperatures[i]} K')
    ax.set_ylabel(r'Scattering phase space ($\times\mathregular{10^3}$ THz$^\mathregular{-1}$)',fontsize=14)
    ax.set_xlabel('Frequency (THz)')
    ax.set_xlim([0,11])
    ax.set_ylim([0,12])
    ax.legend(frameon=False,ncol=2,numpoints=1)