import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.colors as mcolors
import scipy
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

params = {"xtick.direction": "out", "ytick.direction": "out"}
plt.rcParams.update(params)
font = {'family' : 'Arial',
        'size': 15}
matplotlib.rc('font', **font)

direc = 'data'

ifcunit = 27.2107/0.529177249**2
ev2J = 1.60217662e-19
ang = 1e-10
amu = 1.6605390666e-27
hbar = 1.05457182e-34
j2cm1= 5.03411657e22
kB = 1.380649e-23

natom = 188
keywd = 'LS'
keywd = 'HS'
gengif = 0
savenm = 0
showtag = 1
nframe = 40
atom_factor = 0.7
bondrad = 0.06
atom_factor_gif = 120
fsz = 15
elev=15
azim=-95 #-100
quivc = (204/255,204/255,0/255)
quicm = 'YlOrRd'
qlw = 1
font = {'family' : 'Arial',
        'size': fsz}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': fsz})

aa = 13.08
bb = 8.86
cc = 18.716

fig = plt.figure(2,figsize=(5,4.7))
ax = fig.add_subplot(111)

ac = {'Fe':'darkorange',
    'N': 'steelblue',
    'C': 'darkgrey',
    'H': 'ivory',
    'B': 'lightpink'
}

arad = {'Fe':126, # 194
        'N':74,
        'C':77,
        'H':46,
        'B':81
}

eledict = {
    'Fe':55.935,
    'N':14.003,
    'C':12,
    'H':1.008, 
    'B':11.009
}
emass = {
    'Fe':55.845,
    'N':14.0067,
    'C':12.011,
    'H':1.00784,
    'B':10.811
}

elec = {
    'Fe':0,
    'N':1,
    'C':2,
    'H':3, 
    'B':4
}

def general_dist(pos1,pos2,vec2):
    dr = 1000
    P2 = np.zeros(pos2.shape)
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                d1 = np.linalg.norm(pos1-pos2+np.dot(np.array([i,j,k]),vec2))
                pos_mean =  (pos1 + pos2 - np.dot(np.array([i,j,k]),vec2))/2.0
                if d1 < dr:
                    dr = d1
                    P2 = - np.dot(np.array([i,j,k]),vec2)
    return dr,pos_mean,P2

def getclusterid(pos,vec,clusteratom): 
    clusterid = np.zeros(pos.shape[0],dtype=int)
    for i in range(pos.shape[0]):
        dr = 1e4
        for j in range(len(clusteratom)):
            dr1,_,tmp = general_dist(pos[i,:],pos[clusteratom[j],:],vec)
            if dr1 < dr:
                dr = dr1
                ic = j
        clusterid[i] = ic
    return clusterid

def getPR(nmode,element,cid):
    #https://alamode.readthedocs.io/en/latest/anphondir/formalism_anphon.html
    num = 0
    den = 0
    num4 = np.zeros((4,))
    den4 = np.zeros((4,))
    avgvec = np.zeros((4,3))
    avgvec_std = np.zeros((4,3))
    cluster_num_atom = np.zeros((4,),dtype=int)
    elecon = 0
    atpr = 0
    for i in range(nmode.shape[0]):
        nm = np.linalg.norm(nmode[i,:])
        num += nm**2 #/emass[element[i]]
        den += nm**4 #/emass[element[i]]**2
        num4[cid[i]] += nm**2
        den4[cid[i]] += nm**4
        avgvec[cid[i],:] += nmode[i,:] #/np.sqrt(emass[element[i]])
        cluster_num_atom[cid[i]] += 1 
        if element[i] == 'Fe':
            elecon += np.sum(nmode[i,:]**2)
            atpr += nm**2
    for i in range(4):
        avgvec[i,:] = avgvec[i,:]/cluster_num_atom[i]
    for i in range(nmode.shape[0]):
        avgvec_std[cid[i],:] = avgvec_std[cid[i],:] + (nmode[i,:]-avgvec[cid[i],:])**2
    for i in range(4):
        avgvec_std[i,:] = avgvec_std[i,:]/cluster_num_atom[i]
    num = num**2
    num4 = num4**2
    den = den*nmode.shape[0]
    den4 = den4*nmode.shape[0]/4

    return num/den,num4/den4,avgvec,avgvec_std,elecon,atpr/den

def getbond(e1,e2,pos,element,vec,rcut,cid,typ):
    drlist = []
    rlist = []
    id1 = []
    id2 = []
    DR = []
    cd1 = []
    cd2 = []

    for i in range(len(pos)-1):
        for j in range(i+1,len(pos)):
            if (element[i] == e1 and element[j] == e2) or (element[i] == e2 and element[j] == e1):
                if (cid[i] == cid[j] and typ == 'intra') or (cid[i]!=cid[j] and typ == 'inter'):
                    dr,pm,P2 = general_dist(pos[i,:],pos[j,:],vec)
                    if dr < rcut and dr > 0.2:
                        drlist.append(dr)
                        rlist.append(np.linalg.norm(pm))
                        id1.append(i)
                        id2.append(j)
                        DR.append(P2)
                        cd1.append(cid[i])
                        cd2.append(cid[j])
    return np.array(drlist),np.array(rlist),np.array(id1,dtype=int),np.array(id2,dtype=int),np.array(DR),np.array(cd1,dtype=int),np.array(cd2,dtype=int)


omega_all = [] 
vmesh_all = []
ms = 50

def getbe(omega,T):
    return 1/(np.exp(hbar*omega*1e12*2*np.pi/kB/T)-1)
def getC(omega1,omega2,T):
    return hbar*omega1*omega2/T*(getbe(omega1,T)-getbe(omega2,T))/(omega1-omega2)
#  Save the variables to a binary file
def save_geometry_data(filename, natom, pos1, vec1, mass, element, elenum):
    np.savez_compressed(filename, 
                       natom=natom,
                       pos1=pos1, 
                       vec1=vec1,
                       mass=mass,
                       element=element,
                       elenum=elenum)

# Load the variables from the binary file
def load_geometry_data(filename):
    data = np.load(filename, allow_pickle=True)
    natom = data['natom'].item()  # .item() converts 0-d array to scalar
    pos1 = data['pos1']
    vec1 = data['vec1']
    mass = data['mass']
    element = data['element'].tolist()  # Convert back to list
    elenum = data['elenum']
    return natom, pos1, vec1, mass, element, elenum

# Save hessian matrix to binary file
def save_hessian_data(filename, hessian, nhessian):
    np.savez_compressed(filename,
                       hessian=hessian,
                       nhessian=nhessian)

# Load hessian matrix from binary file
def load_hessian_data(filename):
    data = np.load(filename)
    hessian = data['hessian']
    nhessian = data['nhessian'].item()
    return hessian, nhessian

# Load charges from binary file
def load_charges_data(filename):
    data = np.load(filename)
    charges = data['charges']
    return charges

omegaall1 = []
rtall = []
evall = np.zeros((2,natom*3,natom*3),dtype=complex)
for ikwd,keywd in enumerate(['LS','HS']):
    P1 = [[1,0,0],[0,1,0],[0,0,1]]
    natom1,pos1,vec1,mass,element,elenum = load_geometry_data(os.path.join(direc,keywd+'_geometry.npz'))
    posij = np.zeros((natom,natom,3))
    ucij = np.zeros((natom,natom,3))
    for i in range(natom): 
        for j in range(natom):
            dr = 1e10
            for k in range(-1,2):
                for l in range(-1,2):
                    for m in range(-1,2):
                        dist0 = pos1[j,:] - pos1[i,:]+np.dot(np.array([k,l,m]),vec1)
                        dist = dist0[0]**2+dist0[1]**2+dist0[2]**2
                        if dist<=dr:
                            dr = dist 
                            ucij[i,j,:] = np.dot(np.array([k,l,m]),vec1)
                            posij[i,j,:] = dist0
                        
    cid1 = getclusterid(pos1,vec1,[0,1,2,3])

    elist = ['Fe-N','N-N','C-H','N-C','N-B','B-H']
    rcutlist = [2.5,2,2.4,1.8,2.2,1.9]
    bondtype = 'intra'
    atoma = []
    atomb = []
    cda = []
    cdb = []
    DR = []
    for ie,el in enumerate(elist):
        e1 = el.split('-')[0]
        e2 = el.split('-')[1]
        dr1,r1,id1,jd1,Rj,ca,cb = getbond(e1,e2,pos1,element,vec1,rcutlist[ie],cid1,bondtype)
        atoma.append(id1)
        atomb.append(jd1)
        DR.append(Rj)
        cda.append(ca)
        cdb.append(cb)
    atoma = np.concatenate(atoma,axis=0)
    atomb = np.concatenate(atomb,axis=0)
    DR = np.concatenate(DR,axis=0)
    cda = np.concatenate(cda,axis=0)
    cdb = np.concatenate(cdb,axis=0)

    elist = ['N-H']
    rcutlist = [3.1]
    bondtype = 'inter'
    atomha = []
    atomhb = []
    ha = []
    hb = []
    hDR = []
    for ie,el in enumerate(elist):
        e1 = el.split('-')[0]
        e2 = el.split('-')[1]
        dr1,r1,id1,jd1,Rj,ca,cb = getbond(e1,e2,pos1,element,vec1,rcutlist[ie],cid1,bondtype)
        atomha.append(id1)
        atomhb.append(jd1)
        hDR.append(Rj)
        ha.append(ca)
        hb.append(cb)
    atomha = np.concatenate(atomha,axis=0)
    atomhb = np.concatenate(atomhb,axis=0)
    hDR = np.concatenate(hDR,axis=0)
    ha = np.concatenate(ha,axis=0)
    hb = np.concatenate(hb,axis=0)

    hessian, nhessian = load_hessian_data(os.path.join(direc,keywd+'_hessian.npz')) 
    
    # intramolecular and intermolecular force constants 
    hessian1 = np.zeros((nhessian,nhessian))
    hessian2 = np.zeros((nhessian,nhessian))
    for i in range(nhessian):
        for j in range(nhessian):
            if cid1[i//3] == cid1[j//3]:
                hessian1[i,j] = hessian[i,j]
            else:
                hessian2[i,j] = hessian[i,j]
    for i in range(3):
        for j in range(3):
            for k in range(natom):
                hessian[3*k+i,3*k+j] -= np.sum(hessian[3*k+i,j::3])
                hessian1[3*k+i,3*k+j] -= np.sum(hessian1[3*k+i,j::3])
                hessian2[3*k+i,3*k+j] -= np.sum(hessian2[3*k+i,j::3])
    dyn = np.zeros((nhessian,nhessian),dtype=complex)
    dyn1 = np.zeros((nhessian,nhessian),dtype=complex)
    dyn2 = np.zeros((nhessian,nhessian),dtype=complex)
    x0 = np.zeros((nhessian,),dtype=complex)
    x1 = np.zeros((nhessian,),dtype=complex)
    x2 = np.zeros((nhessian,),dtype=complex)
    for i in range(nhessian):
        for j in range(nhessian):
            dyn1[i,j] = hessian1[i,j]/np.sqrt(mass[i//3]*mass[j//3])
            dyn2[i,j] = hessian2[i,j]/np.sqrt(mass[i//3]*mass[j//3])
            dyn[i,j] = hessian[i,j]/np.sqrt(mass[i//3]*mass[j//3])
    eigenvalues, eigenvectors = np.linalg.eig(dyn) 
    
    idx = eigenvalues.argsort()   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    for i in range(nhessian):
        x1[i] = eigenvectors[:,i].conj().T@(dyn1@eigenvectors[:,i])/(eigenvectors[:,i].conj().T@eigenvectors[:,i])/eigenvalues[i]
        x2[i] = eigenvectors[:,i].conj().T@(dyn2@eigenvectors[:,i])/(eigenvectors[:,i].conj().T@eigenvectors[:,i])/eigenvalues[i]
    omega = np.sqrt(eigenvalues[:]*ifcunit/amu*ev2J/ang**2)/1e12/np.pi/2 
    # vel mat
    kxyz = np.array([0,0,0])
    M = np.zeros((natom*3,natom*3),dtype=complex)
    V1 = np.zeros((natom*3,natom*3),dtype=complex)
    V2 = np.zeros((natom*3,natom*3),dtype=complex)
    V3 = np.zeros((natom*3,natom*3),dtype=complex)
    for j in range(natom*3):
        for k in range(natom*3):
            M[j,k] = hessian[j,k]/np.sqrt(mass[j//3]*mass[k//3])*np.exp(1j*np.dot(kxyz,posij[j//3,k//3,:]))
            V1[j,k] = hessian[j,k]/np.sqrt(mass[j//3]*mass[k//3])/2*np.exp(1j*np.dot(kxyz,posij[j//3,k//3,:]))*(-1)*posij[j//3,k//3,0]
            V2[j,k] = hessian[j,k]/np.sqrt(mass[j//3]*mass[k//3])/2*np.exp(1j*np.dot(kxyz,posij[j//3,k//3,:]))*(-1)*posij[j//3,k//3,1]
            V3[j,k] = hessian[j,k]/np.sqrt(mass[j//3]*mass[k//3])/2*np.exp(1j*np.dot(kxyz,posij[j//3,k//3,:]))*(-1)*posij[j//3,k//3,2]
    V1 = eigenvectors.conj().T@(V1@eigenvectors)
    V2 = eigenvectors.conj().T@(V2@eigenvectors)
    V3 = eigenvectors.conj().T@(V3@eigenvectors)
    evall[ikwd] = eigenvectors[:,:]

    Ttr = 331
    for j in range(3*natom):
        for k in range(3*natom):
            delta = 1/np.sqrt(omega[j]*omega[k])
            V1[j,k] = (V1[j,k]/np.sqrt(omega[j]*omega[k])*ifcunit/amu*ev2J/ang**1/2*np.pi/1e12/2)**2*2*np.pi*1e12*getC(omega[j],omega[k],Ttr)/(vec1[0,0]*vec1[1,1]*vec1[2,2])/1e-30\
            *delta/(delta**2+(omega[j]-omega[k])**2)*1e-12
            V2[j,k] = V2[j,k]/np.sqrt(omega[j]*omega[k])*ifcunit/amu*ev2J/ang**1/2*np.pi/1e12/2
            V3[j,k] = V3[j,k]/np.sqrt(omega[j]*omega[k])*ifcunit/amu*ev2J/ang**1/2*np.pi/1e12/2
    vmax = 2e-3
    aa,bb = np.meshgrid(omega[3:],omega[3:])
    idxx = 203
    vmesh_all.append(abs(V1[3:idxx,3:idxx].real))
    
    plt.figure(1)
    for i in range(nhessian):
        plt.plot([omega[i],omega[i]],[1,2],c='C'+str(ikwd))
    plt.xlim([-1,12])
    plt.xlabel('Frequency (THz)')
    plt.yticks([])
    
    def monoExp(x, m, t, b,c,m1,t1,c1):
        return m * np.exp(-t * (x-c)) + b+m1 * np.exp(-t1 * (x-c1))
    p0 = (1, 1, 0.1,0.1,2,0.05,0.1) # start with values near those we expect
    idx1 = np.where(omega[:]>12.5)
    id1 = idx1[0][0]
    params, cv = scipy.optimize.curve_fit(monoExp, omega[3:id1], x2[3:id1], p0,maxfev=100000)
    #m, t, b,  c = params
    sampleRate = 20_000 # Hz
    tauSec = (1 / params[1]) / sampleRate

    # determine quality of the fit
    squaredDiffs = np.square(x2[3:] - monoExp(omega[3:], *params))
    squaredDiffsFromMean = np.square(x2[3:] - np.mean(x2[3:]))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")

    cc = sns.color_palette()
    cedge = sns.dark_palette(cc[3*ikwd],reverse=True,n_colors=3)
    #omegaall1.append(np.array(omega[:2:-1],dtype=float))
    omega_fine = np.linspace(omega[3],omega[id1],300)
    omegaall1.append(np.array(omega_fine,dtype=float))


    if ikwd == 0:    
        marker = 'o'
    else:
        marker='D'
    ax.scatter(omega[:2:-1],x2[:2:-1]*100,ms,color=cedge[0],label=keywd,marker=marker,zorder=5+ikwd,edgecolors=cedge[1])
   
    ntemp = len(omega[:2:-1])//5
    ytemp = np.zeros((ntemp,))
    xtemp2 = omega[:2:-1]
    xtemp = np.zeros((ntemp,))
    ytemp2 = x2[:2:-1]
    for itemp in range(ntemp):
        xtemp[itemp] = np.mean(xtemp2[itemp*5:(itemp+1)*5])
        ytemp[itemp] = np.mean(ytemp2[itemp*5:(itemp+1)*5])*100


    rtall.append(np.array(monoExp(omega_fine, *params)*100,dtype=float))    
    ax.set_xlim([0.0,5.5])
    ax.tick_params(axis='x', which='minor', bottom=True)

    ax.set_ylabel(r'Intermolecular contribution $\eta$ (%)')
    ax.set_xlabel('Frequency (THz)')
    ax.legend(frameon=False,handletextpad=0.1)
    
    omega_all.append(omega)
    calph = 0
    nk1 = 20
    if calph == 1:
        #K-points in fractional coordinates
        #G     :    0.0000     0.0000     0.0000 
        #R     :    0.5000     0.5000     0.5000 
        #S     :    0.5000     0.5000     0.0000 
        #T     :    0.5000     0.0000     0.5000 
        #U     :    0.0000     0.5000     0.5000 
        #X     :    0.0000     0.5000     0.0000 
        #Y     :    0.5000     0.0000     0.0000 
        #Z     :    0.0000     0.0000     0.5000 
        klabels = ['G','R','S','T','U','X','Y','Z']
        kfrac = np.array([[0,0,0],[0.5,0.5,0.5],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0,0.5,0],[0.5,0,0],[0,0,0.5]])
        klabels = ['G','Z']
        kfrac = np.array([[0,0,0],[0,0,0.5]])

        plt.figure(4)
        nk = len(klabels)
        
        kabs = []
        labelk = []
        klen = 0
        icount = 0
        kxyz0 = np.zeros((3,))
        recvec = 2*np.pi*np.transpose(np.linalg.inv(vec1))
        nkpt = nk1*(nk-1)+1
        phdata =np.zeros((nkpt,3*natom))

        omegaall = np.zeros((nhessian,nk))
        for ip in range(nk-1):
            if ip == nk-2:
                kpt1 = np.linspace(kfrac[ip,0],kfrac[ip+1,0],nk1+1)
                kpt2 = np.linspace(kfrac[ip,1],kfrac[ip+1,1],nk1+1)
                kpt3 = np.linspace(kfrac[ip,2],kfrac[ip+1,2],nk1+1) 
            else:
                kpt1 = np.linspace(kfrac[ip,0],kfrac[ip+1,0],nk1,endpoint=False)
                kpt2 = np.linspace(kfrac[ip,1],kfrac[ip+1,1],nk1,endpoint=False)
                kpt3 = np.linspace(kfrac[ip,2],kfrac[ip+1,2],nk1,endpoint=False)
            for ik in range(len(kpt1)):
                print((ip*nk1+ik)/nkpt)
                kxyz = np.dot([kpt1[ik],kpt2[ik],kpt3[ik]],recvec)
                klen = klen + np.linalg.norm(kxyz-kxyz0)
                kabs.append(klen)

                V1 = np.zeros((natom*3,natom*3),dtype=complex)
                V2 = np.zeros((natom*3,natom*3),dtype=complex)
                V3 = np.zeros((natom*3,natom*3),dtype=complex)
                #kxyz = np.array([3/10,3/10,3/10])
                #kxyz = np.array([0,0,i/(nk-1)*1*np.pi/vec1[2,2]])
                M = np.zeros((3*natom,3*natom),dtype=complex)
                for j in range(natom*3):
                    for k in range(natom*3):
                        M[j,k] += hessian[j,k]/np.sqrt(mass[j//3]*mass[k//3])*np.exp(1j*np.dot(kxyz,posij[j//3,k//3,:]))
                        V1[j,k] = hessian[j,k]/np.sqrt(mass[j//3]*mass[k//3])*1j*posij[j//3,k//3,0]*np.exp(1j*np.dot(kxyz,posij[j//3,k//3,:]))
                        V2[j,k] = hessian[j,k]/np.sqrt(mass[j//3]*mass[k//3])*1j*posij[j//3,k//3,1]*np.exp(1j*np.dot(kxyz,posij[j//3,k//3,:]))
                        V3[j,k] = hessian[j,k]/np.sqrt(mass[j//3]*mass[k//3])*1j*posij[j//3,k//3,2]*np.exp(1j*np.dot(kxyz,posij[j//3,k//3,:]))
                eigs, eigenvectors = np.linalg.eig(M)            
                V1 = eigenvectors.conj().T@(V1@eigenvectors)
                V2 = eigenvectors.conj().T@(V2@eigenvectors)
                V3 = eigenvectors.conj().T@(V3@eigenvectors)

                omega = np.sqrt(eigs[:]*ifcunit/amu*ev2J/ang**2)/1e12/np.pi/2 
                eigenvalues = np.zeros(eigs.shape)
                eigenvalues[np.where(eigs.real>=0)] = np.real(omega[np.where(eigs.real>=0)])
                eigenvalues[np.where(eigs.real<0)] = -abs(np.imag(omega[np.where(eigs.real<0)]))
                phdata[icount,:] = np.sort(omega)
                icount = icount + 1
                kxyz0 = kxyz
                if ik == 0:
                    labelk.append(klen)
                elif ik == len(kpt1)-1 and ip == nk-2:
                    labelk.append(klen)
                 
        for ib in range(3*natom):
            plt.plot(kabs,phdata[:,ib],c='C'+str(ikwd),alpha=0.6)
        plt.xticks(labelk,klabels)
        plt.ylim([-0.5,4])
       
    # Read Mulliken charges
    charges = load_charges_data(os.path.join(direc,keywd+'_charges.npz'))

    CC = sns.color_palette("Set2")
    ccc = sns.color_palette()

    mdlist = np.arange(8) #[0,1] #,3,4,6,7,8,11,12,400,401,500]
    imd = 0

xf1 = [2.96,3.39]
yf = [-10,-10]
yf1 = [101,101]
xf2 = [6.157,6.966]


ax.set_xlim([0,11])
ax.set_ylim([-3,100])

fig.subplots_adjust(left=0.175, right=0.95, top=0.94, bottom=0.17)
fig.savefig('inter_vs_intra.png',transparent=True,dpi=600)

fig = plt.figure(3,figsize=(5,4.7))
plt.xlim([0,16])
plt.xlim([0.3,13.5])
plt.ylim([0.6,1.15])
plt.xlabel(r'$\omega_\mathrm{LS}$ (THz)')
plt.ylabel(r'$(\omega_\mathrm{LS}/\omega_\mathrm{HS})^2$')
xxxx = np.linspace(0.3,13.5,500)
yyyy = np.linspace(0,1.2,100)

thz2meV = 4.136
meV = 1.60218e-19/1000
alpha = 1
z = np.empty((len(yyyy), len(xxxx), 4), dtype=float)

ccc = sns.color_palette("icefire",n_colors=5)
rgb = mcolors.colorConverter.to_rgb('C0')
z[:,:,:3] = rgb
T = 331 
nbe = 1/(np.exp(xxxx*thz2meV*meV/kB/T)-1)
nbe = nbe/nbe[0]*alpha

z[:,:,-1] = nbe[None,:] 
ax = plt.gca()
ccc0 = 'C1' 
cedge = sns.dark_palette(ccc0, reverse=True,n_colors=3)
aaa = omega_all[0][:2:-1]
bbb = (omega_all[0][:2:-1]/omega_all[1][:2:-1])**2
ida = np.where(aaa<11)[0]
plt.scatter(aaa[ida],bbb[ida],ms,edgecolors=cedge[1],color=cedge[0],zorder=2,marker='o')
plt.xlim([0.,11.0])
plt.ylim([0.9,2.1])
x = omega_all[0][:2:-1]
y = omega_all[0][:2:-1]/omega_all[1][:2:-1]
idx2 = np.where(x<12.5)
z = np.polyfit(x[idx2],y[idx2], 1)
p = np.poly1d(z)
fig.subplots_adjust(left=0.175, right=0.95, top=0.94, bottom=0.17)

fig.savefig('omega_ratio.png',transparent=True,dpi=600)
print('lowest OP:',omega_all[0][3],omega_all[1][3])
plt.show()
