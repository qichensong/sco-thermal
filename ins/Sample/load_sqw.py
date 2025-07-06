# Set the plot style
import mpl_style
mpl_style.set_style()
import matplotlib.pyplot as plt

# load relavant packages
import numpy as np
import os
import seaborn as sns
from constants import *
from functions import get2theta,readtxtfile
import sqw_plotting,Aqw_plotting,intQ,plotdos,plot_spectral_C

class sqw:
    def __init__(self,wavelength,data_dir='.'):
        self.lambda_neutron = wavelength # wavelength in Angstrom
        self.data_dir = data_dir # directory where data files are stored
        if not os.listdir(self.data_dir):
            print(f"Data folder '{self.data_dir}' is empty.")
            self.files = []
            self.nfiles = 0
        else:
            self.files =[f for f in os.listdir(self.data_dir)]
            print(f"Data files found: {self.files}")
            self.nfiles = len(self.files)
        # integrated dos (unscaled), just a number, initialized to be zero
        self.int_dos = np.zeros((self.nfiles,))

    def read_data(self,file_index=0,temperature=300): 
        self.data, self.data_e, self.elastic, self.x, self.y = readtxtfile(os.path.join(self.data_dir, self.files[file_index])) 
        self.temperature = temperature

    def plot_elastic(self):
        idx = np.where(self.elastic[:,0]>0)
        plt.plot(get2theta(self.y[idx],self.lambda_neutron),self.elastic[idx[0],0])
        
    def read_all_data(self,all_tempeature):
        self.all_data = []
        self.all_data_e = [] # errorbars
        self.all_elastic = []
        self.all_x = []
        self.all_y = []
        self.all_temperature = all_tempeature
        for i in range(self.nfiles):
            print(f"Reading file {i+1}/{self.nfiles}: {self.files[i]}")
            self.read_data(i,self.all_temperature[i])
            self.plot_elastic()
            self.all_data.append(self.data) 
            self.all_data_e.append(self.data_e)
            self.all_elastic.append(self.elastic)
            self.all_x.append(self.x)
            self.all_y.append(self.y)
    
    def plot_sqw(self,save_dir='saved_figures',file_index_input=None): 
        if file_index_input is None:
            file_index = range(self.nfiles)
        else:
            file_index = file_index_input
        # make the save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)
        for i in file_index:
            sqw_plotting.sqw_plotting(i,self.all_data[i],self.all_x[i],self.all_y[i]) 
            plt.savefig(os.path.join(save_dir, f'sqw_{self.all_temperature[i]}K.png'), dpi=300)
    def get_gdos(self,vol,natom,save_dir='saved_figures',file_index_input=None):
        self.all_gdos = []
        self.all_energy = []
        self.all_scaling = []
        if file_index_input is None:
            file_index = range(self.nfiles)
        else:
            file_index = file_index_input
        # make the save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)
        for i in file_index:
            scaling, energy, gdos, self.int_dos[i] = intQ.intQ(i,self.all_data[i],self.all_data_e[i],self.all_y[i],self.all_x[i],self.all_temperature[i],vol[i],natom)
            self.all_gdos.append(gdos)
            self.all_energy.append(energy)
            self.all_scaling.append(scaling)

    def plot_aqw(self,save_dir='saved_figures',file_index_input=None):
        if file_index_input is None:
            file_index = range(self.nfiles)
        else:
            file_index = file_index_input
        # make the save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)
        for i in file_index:
            if self.int_dos[i] == 0:
                print('Q integration has not been performed yet. Please run get_gdos() first.')
            else:
                Aqw_plotting.Aqw_plotting(i,self.all_data[i],self.all_x[i],self.all_y[i],self.all_scaling[i],self.all_temperature[i])
                plt.savefig(os.path.join(save_dir, f'Aqw_{self.all_temperature[i]}K.png'), dpi=300)
                plt.close()
    
    def plot_gdos(self,save_dir='saved_figures',file_index_input=None):
        if file_index_input is None:
            file_index = [i for i in range(self.nfiles) if self.int_dos[i] != 0]
        else:
            file_index = file_index_input
        # make the save directory if it does not exist
        file_index = np.array(file_index)
        os.makedirs(save_dir, exist_ok=True)
        plotdos.plotdos(np.array(self.all_energy)[file_index],np.array(self.all_gdos)[file_index],np.array(self.all_temperature)[file_index])
        plt.savefig(os.path.join(save_dir, 'gdos_T.png'), dpi=300)
    
    def plot_spectral_C(self,save_dir='saved_figures',file_index_input=None):
        if file_index_input is None:
            file_index = [i for i in range(self.nfiles) if self.int_dos[i] != 0]
        else:
            file_index = file_index_input
        # make the save directory if it does not exist
        file_index = np.array(file_index)
        os.makedirs(save_dir, exist_ok=True)
        plot_spectral_C.plot_spectral_C(np.array(self.all_energy)[file_index],np.array(self.all_gdos)[file_index],np.array(self.all_temperature)[file_index])
        plt.savefig(os.path.join(save_dir, 'spectral_C_T.png'), dpi=300)
        
        
sqw_instance = sqw(4.69,'data') # neutron wavelength and data directory 
#sqw_instance.read_data(0)  # Read the first file
sqw_instance.read_all_data([250,300,350,375,400])  # Read all files
sqw_instance.plot_sqw()  # Plot S(q,w)
# unit cell volume in Angstrom^3 at listed temperatures
vol = np.array([2063.284144, 2074.812745, 2175.979671, 2178.346967, 2178.346967])
natom = 188
sqw_instance.get_gdos(vol,natom)  # Calculate the integrated generalized density of states
sqw_instance.plot_aqw()  # Plot weighted S(q,w), denoted as A(q,w)
sqw_instance.plot_gdos()  # Plot the generalized density of states
sqw_instance.plot_spectral_C()  # Plot the spectral specific heat
plt.show()

