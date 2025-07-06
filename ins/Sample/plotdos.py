import matplotlib.pyplot as plt
from constants import meV2thz

def plotdos(all_energy, all_gdos, all_temperature):
    fig = plt.figure(21,figsize=(5,4.4))
    ax = fig.add_subplot(111)
    for i in range(len(all_temperature)):
        ax.plot(all_energy[i]*meV2thz, all_gdos[i]/1e27/meV2thz, label=f'{all_temperature[i]} K')
    ax.legend(frameon=False)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel(r'GDOS (THz$\mathregular{^{-1}}$nm$\mathregular{^{-3}}$)')
    ax.set_xlim([0, 13])
    ax.set_ylim([0, 13])