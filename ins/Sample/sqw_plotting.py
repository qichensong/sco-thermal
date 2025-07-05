import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from constants import meV2thz

# function to plot the S(q,w) data
def sqw_plotting(file_index,data,x,y):
    fig = plt.figure(91+file_index,figsize=(5,4.4))
    ax = fig.add_subplot(111)
    xel, yel = np.meshgrid(x, y)
    data_pos = data
    data_pos[data_pos<0] = 1e-6
    cmp = 'nipy_spectral'
    data_pos = data_pos + 1e-10
    pcm = plt.pcolormesh(yel,-xel*meV2thz,data_pos,norm=colors.LogNorm(vmin=10**(-4), vmax=10**2),cmap=cmp,shading='auto')
    llmt = 10**(-4)
    hlmt = 10**2
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
    axins.tick_params(which="major", axis="x", direction="out")
    cbar.ax.set_xticklabels([r'$\mathregular{10^{-4}}$', r'$\mathregular{10^{2}}$'])  # horizontal colorbar
    cbar.ax.tick_params(labelsize=13)
    axins.tick_params(which="major", axis="x", direction="out",colors='white')
    axins.tick_params(which="minor", axis="x", direction="out",colors='white',labelbottom=False)
    axins.tick_params(axis='x',color='white')
    axins.tick_params(axis='y',color='white')
    cbar.outline.set_color('white')
    axins.spines['bottom'].set_color('white')
    axins.spines['top'].set_color('white') 
    axins.spines['right'].set_color('white')
    axins.spines['left'].set_color('white')
    ax.set_xlabel(r'$Q$ ($\mathregular{\AA^{-1}}$)')
    ax.set_ylabel('Energy transfer (THz)')
    ax.set_ylim([-0.5,12])
    ax.set_xlim([0,6])