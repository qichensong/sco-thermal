import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from constants import meV2thz

# plot the weighted S(q,w) data, namely, A(q,w)
def Aqw_plotting(file_index,data,x,y,scaling=1,T=300):
    fig = plt.figure(31+file_index,figsize=(5,4.4))
    ax = fig.add_subplot(111)
    cmp = 'nipy_spectral'
    llmt = 0
    hlmt = 1e-3
    xv, yv = np.meshgrid(y, x) 
    data_scaled = scaling*data.T*abs(yv)/xv**2/1e30*(1-np.exp(-abs(yv)/(T*25.7/298)))
    pcm = ax.pcolormesh(xv,-yv*meV2thz,data_scaled,vmin=llmt,vmax=hlmt,cmap=cmp,shading='auto')
    ax.set_ylim([0.2,12])
    ax.set_xlim([0,5.5])
    pcm.set_clim(llmt,hlmt)
    axins = inset_axes(
    ax,
    width="15%",  # width: 5% of parent_bbox width
    height="5%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(0.78, 0.12, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
    ) 
    cbar = fig.colorbar(pcm, cax=axins,ax = ax,orientation="horizontal",ticks=[llmt,hlmt]) 
    axins.tick_params(which="major", axis="x", direction="out",colors='white')
    axins.tick_params(axis='x',color='white')
    axins.tick_params(axis='y',color='white')
    cbar.outline.set_color('white')
    axins.spines['bottom'].set_color('white')
    axins.spines['top'].set_color('white') 
    axins.spines['right'].set_color('white')
    axins.spines['left'].set_color('white')
    cbar.ax.set_xticklabels(['0', '0.001'],color='white')  # horizontal colorbar
    cbar.ax.tick_params(labelsize=13)
    ax.set_xlabel(r'$Q$ ($\mathregular{\AA^{-1}}$)')
    ax.set_ylabel('Energy (THz)')
