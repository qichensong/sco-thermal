import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LightSource
import matplotlib
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')
from get_inter_vs_intra_from_hessian import load_geometry_data,load_hessian_data

direc = 'data'
params = {"xtick.direction": "out", "ytick.direction": "out"}
plt.rcParams.update(params)

font = {'family' : 'Arial',
        'size': 15}
matplotlib.rc('font', **font)

KBT = 4.11e-21/298*331
m0 =  1.66053906660e-27
w0 = 29.979000e9*2*np.pi
aufactor = 3.1

lscale = np.sqrt(KBT/m0)/w0/1e-10 #*np.sqrt(188*3) # ang

ifcunit = 27.2107/0.529177249**2
ev2J = 1.60217662e-19
ang = 1e-10
amu = 1.6605390666e-27
hbar = 1.05457182e-34
j2cm1= 5.03411657e22


gengif = 1
nframe = 40
atom_factor = 0.55
bondrad = 0.06
atom_factor_gif = 120
fsz = 15
elev=15
azim=-95
quivc = (204/255,204/255,0/255)
quicm = 'YlOrRd'
quicm = 'Reds'
quicm = 'autumn'
qlw = 4
qscale = 38

qlw = 1
font = {'family' : 'Arial',
        'size': fsz}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': fsz})

aa = 13.08
bb = 8.86
cc = 18.716

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

# Compute the bond length
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

# Find out the molecule that the atom belongs to
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
# Search bonding neighbors
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
# Make the movie
def visualize_mode_anime(iframe,nmode,pos1,element,vec,fre,atm1,atm2,DR,cda,cdb,clusters,eleselect,hatm1,hatm2,hDR,ha,hb):
    ax.cla()  
    frac = 8*np.cos(iframe/nframe*np.pi*2) 
    rep = np.zeros((natom,))
    hrep = np.zeros((natom,))    
    for i in range(natom):
        for ic in clusters:
            if cid1[i] == ic and element[i] in eleselect:
                cedge = sns.dark_palette(ac[element[i]], reverse=True,n_colors=3)
                cedge = sns.dark_palette(ac[element[i]], reverse=True,n_colors=3)
                ax.scatter(pos1[i,0]+nmode[i,0]*frac,pos1[i,1]+nmode[i,1]*frac,pos1[i,2]+nmode[i,2]*frac,s=atom_factor_gif*arad[element[i]]**2/1e4,color=ac[element[i]],edgecolor=cedge[1])
    for i in range(len(atm1)):
        for ic in clusters:
            if cda[i] == ic and cdb[i] == ic and element[atm1[i]] in eleselect and element[atm2[i]] in eleselect:
                ax.plot3D([pos1[atm1[i],0]+nmode[atm1[i],0]*frac,pos1[atm2[i],0]+DR[i,0]+nmode[atm2[i],0]*frac],
                          [pos1[atm1[i],1]+nmode[atm1[i],1]*frac,pos1[atm2[i],1]+DR[i,1]+nmode[atm2[i],1]*frac], 
                          [pos1[atm1[i],2]+nmode[atm1[i],2]*frac,pos1[atm2[i],2]+DR[i,2]+nmode[atm2[i],2]*frac],color='grey')
                if np.linalg.norm(DR[i,:]) > 0: # and rep[atm2[i]] == 0:
                    rep[atm2[i]] += 1
                    cedge = sns.dark_palette(ac[element[atm2[i]]], reverse=True,n_colors=3)
                    ax.scatter(pos1[atm2[i],0]+DR[i,0]+nmode[atm2[i],0]*frac,
                               pos1[atm2[i],1]+DR[i,1]+nmode[atm2[i],1]*frac,
                               pos1[atm2[i],2]+DR[i,2]+nmode[atm2[i],2]*frac,
                               s=atom_factor_gif*arad[element[atm2[i]]]**2/1e4,color=ac[element[atm2[i]]],edgecolor=cedge[1])
    for i in range(len(hatm1)):
        ax.plot3D([pos1[hatm1[i],0]+nmode[hatm1[i],0]*frac,pos1[hatm2[i],0]+hDR[i,0]+nmode[hatm2[i],0]*frac],
                  [pos1[hatm1[i],1]+nmode[hatm1[i],1]*frac,pos1[hatm2[i],1]+hDR[i,1]+nmode[hatm2[i],1]*frac], 
                  [pos1[hatm1[i],2]+nmode[hatm1[i],2]*frac,pos1[hatm2[i],2]+hDR[i,2]+nmode[hatm2[i],2]*frac],color='royalblue',linestyle='--',lw=0.7)
        if np.linalg.norm(hDR[i,:]) > 0: # and hrep[hatm2[i]] == 0:
            hrep[hatm2[i]] += 1
            cedge = sns.dark_palette(ac[element[hatm2[i]]], reverse=True,n_colors=3)
            ax.scatter(pos1[hatm2[i],0]+hDR[i,0]+nmode[hatm2[i],0]*frac,
                       pos1[hatm2[i],1]+hDR[i,1]+nmode[hatm2[i],1]*frac,
                       pos1[hatm2[i],2]+hDR[i,2]+nmode[hatm2[i],2]*frac,
                       s=atom_factor_gif*arad[element[hatm2[i]]]**2/1e4,color=ac[element[hatm2[i]]],edgecolor=cedge[1])

    limconst = 2
    ax.set_xlim([-aa/limconst,aa/limconst])
    ax.set_ylim([-bb/limconst,bb/limconst])
    ax.set_zlim([-cc/limconst,cc/limconst])
    ax.set_box_aspect(aspect=(aa, bb, cc))
    ls = LightSource(azdeg=0, altdeg=65)
    ax.set_title(str(round(np.real(fre),2))+r' $\mathrm{cm^{-1}}$')

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('a',labelpad=16)
    ax.set_ylabel('b',labelpad=16)
    ax.set_zlabel('c',labelpad=16)
    ax.tick_params(axis='z', which='major', pad=9)
    ax.set_yticks(ticks=[-4,-2,0,2,4])   
    


for ikwd,keywd in enumerate(['LS','HS']):
    P1 = [[1,0,0],[0,1,0],[0,0,1]]
    natom,pos1,vec1,mass,element,elenum = load_geometry_data(os.path.join(direc,keywd+'_geometry.npz'))
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
    for i in range(3):
            for j in range(3):
                for k in range(natom):
                    hessian[3*k+i,3*k+j] -= np.sum(hessian[3*k+i,j::3])
    nmode_all = np.zeros((nhessian,nhessian//3,3)) 
    dyn = np.zeros((nhessian,nhessian),dtype=complex)
    for i in range(nhessian):
        for j in range(nhessian): 
            dyn[i,j] = hessian[i,j]/np.sqrt(mass[i//3]*mass[j//3])
    eigenvalues, eigenvectors = np.linalg.eig(dyn) 
        
    idx = eigenvalues.argsort()   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    for i in range(nhessian):
        for j in range(nhessian//3):
            nmode_all[i,j,:] = eigenvectors[3*j:3*j+3,i]/np.sqrt(mass[j])*aufactor
    idx = eigenvalues.argsort()   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    for i in range(nhessian):
        for j in range(nhessian//3):
            nmode_all[i,j,:] = eigenvectors[3*j:3*j+3,i]/np.sqrt(mass[j])*aufactor
    omega = np.sqrt(eigenvalues[:]*ifcunit/amu*ev2J/ang**2)/1e12/np.pi/2 
    cm2THz = 1/33.35641
    freq = omega/cm2THz
    mdlist = [3,4]
    for imd in mdlist:
        if gengif == 1:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d',facecolor='whitesmoke')
            ani = animation.FuncAnimation(fig, visualize_mode_anime, 
            fargs=(nmode_all[imd],pos1,element,vec1,freq[imd],atoma,atomb,DR,cda,cdb,[0,1,2,3],['Fe','N','C','B','H'],atomha,atomhb,hDR,ha,hb),interval = 10,frames=nframe)
            writergif = animation.PillowWriter(fps=15)
            plt.close(fig)
            ani.save(keywd+'_mode_'+str(imd)+'.gif', writer=writergif) 