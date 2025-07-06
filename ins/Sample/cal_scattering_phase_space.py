import numpy as np
from constants import hbar, kB
from functions import de, deltag
def get_scatt_phase_space(area,freq,sig,T):
    SP1 = np.zeros(freq.shape)
    SP2 = np.zeros(freq.shape)
    f = de(freq,T)
    freq3 = freq
    f3 = f
    for i in range(1,len(freq)):
        freq1 = freq[i]
        temp1 = np.zeros(len(freq))
        temp2 = np.zeros(len(freq))
        for j in range(1,len(freq)): 
            freq2 = freq[j]
            f2 = f[j] 
            temp11 = np.zeros(len(freq))
            temp11 = (f2-f3) *deltag(freq1+freq2-freq3,sig)*area[j]*area
            temp11[0] = 0
            temp1[j] = np.trapz(temp11,freq3)
            temp22 = np.zeros(len(freq))
            temp22 = (f2+f3+1) *deltag(freq1-freq2-freq3,sig)*area[j]*area
            temp22[0] = 0
            temp2[j] = np.trapz(temp22,freq3)
        SP1[i] = np.trapz(temp1,freq)
        SP2[i] = np.trapz(temp2,freq)
    return SP1, SP2