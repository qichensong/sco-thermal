import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.optimize import curve_fit


# Gaussian function
def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

listing = glob.glob(os.path.join('data','*.txt.*'))
nf = len(listing)
cpalette = sns.color_palette("viridis", n_colors=nf)

dq = []
dq1 = []
q = []
width = []
plt.figure(1)
for il in range(nf):
    f = open(listing[il],'r')
    lines = f.readlines()
    f.close()
    nl = len(lines)
    data = []
    for i in range(nl):
        if lines[i].find('Energy') != -1:
            n3 = i
        if lines[i].find('UNKNOWN') != -1:
            q.append(lines[i].split()[-1].split('=')[-1].split('A')[0])
            break
    for i in range(n3):
        line0 = lines[i].split()
        for j in range(3):
            data.append(float(line0[j]))
    data = np.array(data,dtype=float)
    data = data.reshape((n3,3))
    popt, pcov = curve_fit(gauss, data[:, 0], data[:, 1], p0=[np.max(data[:, 1]), 0, 0.2])
    width.append(popt[2])

    plt.plot(data[:,0],data[:,1]+8e-4*il,c=cpalette[il])
    plt.text(0.45,8e-4*il,'$Q_0$ = %s' % q[-1])
    for ie in range(data.shape[0]):
        if abs(data[ie,0])<1e-4:
            i0 = ie
            dq.append(data[ie,1])
            dq1.append(data[ie,2])
            break
dq = np.array(dq,dtype=float)
dq1 = np.array(dq1,dtype=float)
q = np.array(q,dtype=float)
width = np.array(width,dtype=float) * 2.355 # sigma to FWHM
plt.yticks([])
plt.ylabel('$S_\mathrm{vanadium}$($Q = Q_0$,$\omega$) (a.u.)')
plt.xlim([-0.7,0.7])
plt.xlabel('Energy $\hbar\omega$ (meV)')
plt.figure(2)
plt.plot(q,width,'o-')
plt.ylabel('FWHM (meV)')
plt.xlabel('$Q$ ($\mathrm{\AA}^{-1}$)')
plt.ylim([0.075,0.2])
plt.figure(3)
plt.plot(q,dq,'o-',label='vanadium')
plt.plot(q,dq1,'o-',label='empty can (for background subtraction)')
plt.legend(frameon=False)
plt.xlabel('$Q$ ($\mathrm{\AA}^{-1}$)')
plt.ylabel('$S$($Q$,$\omega = 0$) (a.u.)')
plt.show()
#plt.savefig('elastic.png')



