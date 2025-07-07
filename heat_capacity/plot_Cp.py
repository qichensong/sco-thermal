import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from io import StringIO
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline,CubicSpline

savetag = 1

direc = 'saved_figures'

# plot and color style
cc = sns.color_palette()
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='6'
plt.rcParams.update({'font.size': 15})
font = {'family' : 'Arial'}
matplotlib.rc('font', **font)

rhoT = np.array([[-173.15,1.58131],
[-123.15,1.57648],
[-73.15,1.57128],
[-23.15,1.57019],
[26.85,1.560791704],
[36.85,1.556761567],
[51.85,1.552300246],
[56.85,1.545036987],
[61.85,1.497980888],
[66.85,1.492008017],
[76.85,1.488424966],
[81.85,1.486022259],
[86.85,1.482877424]])
rhoT[:,0] = rhoT[:,0] + 273.15

# Sample heat capacity using PPMS
data = """
1.780987967	0.000126745
1.883570033	0.000143806
1.9826492	0.000161015
2.085694967	0.000180171
2.198367867	0.000203506
2.314960067	0.000230201
2.439562367	0.000261806
2.573012767	0.000299291
2.715939833	0.000343955
2.868405333	0.000396654
3.0317405	0.000460888
3.205394667	0.000537694
3.390525167	0.00062958
3.588130967	0.000740016
3.798642867	0.000873261
4.024264767	0.001037251
4.2613939	0.001233362
4.5136768	0.001467078
4.781437767	0.001748023
5.0665885	0.002089151
5.369604133	0.002492365
5.690837767	0.00296475
6.031964767	0.003530184
6.392772667	0.004200311
6.775011933	0.00498204
7.1795788	0.005887296
7.608258033	0.006920812
8.065076733	0.008108019
8.5449101	0.009474461
9.052140267	0.011001643
9.592827033	0.012710499
10.16379033	0.014609927
10.76861	0.016736689
11.407651	0.019094992
12.08427567	0.021675549
12.800809	0.024487286
13.55912267	0.027571917
14.36110067	0.030962019
15.25212233	0.034760963
16.14331133	0.038635157
17.091834	0.042931579
18.09450133	0.047500197
19.15592233	0.052367281
20.279867	0.057593944
21.472359	0.06321615
22.73603933	0.069143963
24.072317	0.075426983
25.48857767	0.082129682
26.98661233	0.089249372
28.57368433	0.096641295
30.25513933	0.104460881
32.034804	0.112792726
33.92114133	0.121706204
35.91859633	0.13106381
38.02939567	0.140529328
40.262434	0.150259456
42.62715567	0.1606773
45.130114	0.17174478
47.78138667	0.182463847
50.58867167	0.194363187
53.57250333	0.206703269
56.710323	0.219680439
60.02762867	0.233311016
63.536736	0.247233808
67.249147	0.261613691
71.17758667	0.276843466
75.336976	0.291621342
79.74520333	0.304979122
84.41626333	0.319682137
89.36310167	0.336979493
94.59490367	0.356418858
100.10899	0.377205285
110.1261233	0.408586316
120.08811	0.434833691
130.12238	0.463939615
140.1528667	0.493224482
150.18373	0.519870802
160.20733	0.548565427
170.2090433	0.576634954
180.2134567	0.602954858
190.22998	0.631367291
200.2576333	0.661764178
210.2913667	0.69236822
220.3511233	0.727596103
230.40061	0.759855039
240.41297	0.793077914
250.4033067	0.828086993
260.3918333	0.868507192
270.37905	0.915342751
280.35681	0.976011885
290.3318867	1.044895602
300.29573	1.114132643
310.3899333	1.217560239
315.4735967	1.312792353
320.3683267	1.460022939
322.4527133	1.567640745
324.3801933	1.712968748
326.3636267	1.958123171
328.3326933	2.368916711
330.2492767	2.735916843
332.16869	2.622392394
334.2065133	2.211995419
336.2236667	1.898110285
338.21959	1.705086521
340.21206	1.580815764
345.2799533	1.416056183
350.1338767	1.346677023
360.1305967	1.287348697
369.8814033	1.281992328
379.8439833	1.292768727
384.61873	1.301288127
"""

def cdebye(T,thetaD):
    R = 8.31446261815324 
    c = np.zeros(T.shape)
    for i in range(len(T)):
        t = thetaD/T[i]
        x = np.linspace(0,t,10000)
        intg = x**4*np.exp(x)/(np.exp(x)-1)**2
        intg[0] = 0
        c[i] = 9*R*(t**-3)*np.trapz(intg,x)*47
    return c 
def cdebye_linear(T,D,thetaD,A,B):
    return D*cdebye(T,thetaD)+A*T+B
def resi(x,T,C):
    y = x[0]*cdebye(T,x[1])-C
    return y
def resiL(x,T,C):
    y = cdebye_linear(T,x[0],x[1],x[2],x[3])-C
    return y
def resiL_scale(x,T,C,D,thetaD,A,B):
    y = x*cdebye_linear(T,D,thetaD,A,B)-C
    return y

# Load heat capacity data
df = pd.read_excel(os.path.join('data','cp comparisons.xlsx'))
df1 = pd.read_excel(os.path.join('data','SCOCpData.xlsx'),skiprows=1)
gmol = 487.88
data = np.genfromtxt(StringIO(data),dtype=float)
idx = np.argsort(data[:,0])
fig0, ax0 = plt.subplots(figsize=(5,4.4))
ccc = sns.color_palette('Paired')

iskip = 65
iskip2 = 71
mss = 18

c2 = sns.color_palette('Paired')
ax0.scatter(data[iskip2:,0],data[iskip2:,1]*gmol,mss,edgecolors=c2[1],zorder=15,alpha=0.9,facecolors=c2[0],label='PPMS')
plt.scatter(data[iskip:iskip2:2,0],data[iskip:iskip2:2,1]*gmol,mss,edgecolors=c2[1],zorder=15,alpha=0.9,facecolors=c2[0])
plt.scatter(data[:iskip:3,0],data[:iskip:3,1]*gmol,mss,edgecolors=c2[1],zorder=15,alpha=0.9,facecolors=c2[0])
datasave = data

data = data[idx,:]
data[:,1] = data[:,1]*gmol
iend = 86
print('curve fit')
print('end',data[iend,0])
resi1 = least_squares(resi,[1,200],args=(data[:iend,0],data[:iend,1]),bounds=([0,1],[100,1000]))
print(resi1.x)
resi2 = least_squares(resiL,[1,300,0.5,0],args=(data[:iend,0],data[:iend,1]),bounds=([0,1,0,-500],[1000,1000,1000,1000]))
print(resi2.x)

istart = 108
print('start',data[istart,0])
resi3 = least_squares(resiL,[1,300,0.5,0],args=(data[istart:,0],data[istart:,1]),bounds=([0,1,0,-500],[1000,1000,1000,500]))
print(resi3.x)


Ttr = 331.6
Ttr1 = 329.9
Ttr2 = 336
T1 = np.linspace(data[0,0],Ttr,200)
C1 = cdebye_linear(T1,resi2.x[0],resi2.x[1],resi2.x[2],resi2.x[3])

plt.subplots_adjust(left=0.205, right=0.98, top=0.94, bottom=0.17)
plt.ylabel(r'Molar heat capacity (J mol$\mathregular{^{-1}}$ K$\mathregular{^{-1}}$)')
plt.xlabel('Temperature (K)')
plt.xlim([0,390])
plt.ylim([0,700])
ax0.plot(T1,C1,ls='-',c=cc[0],zorder=9,lw=2,label='Debye-linear fit')
print('linear-in-T:',
resi2.x[0],resi2.x[1],resi2.x[2],resi2.x[3])

T2 = np.linspace(Ttr+0.0001,data[-1,0],200)
C2 = cdebye_linear(T2,resi3.x[0],resi3.x[1],resi3.x[2],resi3.x[3])
ax0.plot(T2,C2,ls='-',c=cc[0],zorder=9,lw=2)
print((C2[0]-C1[-1])/C1[-1]*100,'percent, mass')
xco = df1[df1.columns[0]].values
yco = df1[df1.columns[1]].values
idx_co1 = np.where(xco<275)

handles, labels = plt.gca().get_legend_handles_labels()
if os.name == 'nt':
    order = [0,1]
else:
    order = [1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False,loc='upper left')
newT = np.concatenate((T1,T2[1:]))
newC = np.concatenate((C1,C2[1:]))
bl = np.interp(data[:,0],newT,newC)
idxT = np.where(data[:,0] > 250)
print('dS',np.trapz(data[idxT,1]-bl[idxT],data[idxT,0])/Ttr/gmol*1000)

idxb = np.where((data[:,0] < 255) | (data[:,0]> 370))[0]
x = data[idxb,0]
y = data[idxb,1]/gmol*1000
X_Y_Spline = make_interp_spline(x, y)

T = np.array([100,150,200,250,300,325,330,335,340,350,355,360])
V = np.array([2049.937592,2055.167452,2060.697378,2063.284144,2074.812745,2088.455528,2095.610365,2160.701022,2167.650246,2175.979671,2177.815601,2178.346967])
ag = 6.02214076e23
plt.scatter(data[[iend-1,istart],0],data[[iend-1,istart],1],180,edgecolors=None,facecolors='gray',marker='|',zorder=9)
if savetag:
    plt.savefig(os.path.join(direc,'molar_heat_capacity_parent.png'),dpi=600,transparent=True)

fig0, ax0 = plt.subplots(figsize=(5,4.4))

ax0.scatter(datasave[iskip2:,0],datasave[iskip2:,1]*gmol,mss,edgecolors=c2[1],zorder=1,alpha=0.9,facecolors=c2[0],label=r'Fe[HB(tz)$\mathrm{_3}$]$\mathrm{_2}$, PPMS')
plt.scatter(datasave[iskip:iskip2:2,0],datasave[iskip:iskip2:2,1]*gmol,mss,edgecolors=c2[1],zorder=1,alpha=0.9,facecolors=c2[0])
plt.scatter(datasave[:iskip:3,0],datasave[:iskip:3,1]*gmol,mss,edgecolors=c2[1],zorder=1,alpha=0.9,facecolors=c2[0])
plt.scatter(data[[iend-1,istart],0],data[[iend-1,istart],1],180,edgecolors=None,facecolors='gray',marker='|',zorder=1)
ax0.plot(T1,C1,ls='-',c=cc[0],lw=2,label=r'Fe[HB(tz)$\mathrm{_3}$]$\mathrm{_2}$, Debye-linear model',zorder=0)
ax0.plot(T2,C2,ls='-',c=cc[0],zorder=0,lw=2)
ccc = sns.color_palette('Paired')
cdb = sns.light_palette('midnightblue',3)
cf = cdb[1]
resi4 = least_squares(resiL,[1,300,0.5,0],args=(xco[idx_co1],yco[idx_co1]),bounds=([0,1,0,-500],[1000,1000,1000,1000]))
Ttemp4 = np.linspace(xco[idx_co1][0],xco[idx_co1][-1],200)
print(Ttemp4)
Ctemp4 = cdebye_linear(Ttemp4,resi4.x[0],resi4.x[1],resi4.x[2],resi4.x[3])

ax0.scatter(xco[idx_co1],yco[idx_co1],mss,label=r'Co[HB(tz)$\mathrm{_3}$]$\mathrm{_2}$, PPMS',facecolors=cf,edgecolors='midnightblue',alpha=0.9,zorder=8)

plt.subplots_adjust(left=0.205, right=0.98, top=0.94, bottom=0.17)
plt.ylabel(r'Molar heat capacity (J mol$\mathregular{^{-1}}$ K$\mathregular{^{-1}}$)')
plt.xlabel('Temperature (K)')
plt.xlim([0,390])
plt.ylim([0,700])

ax0.plot(Ttemp4,Ctemp4,label=r'Co[HB(tz)$\mathrm{_3}$]$\mathrm{_2}$, Debye-linear model',c='midnightblue',lw=2,ls='-',zorder=10)

handles, labels = ax0.get_legend_handles_labels()
if os.name == 'nt':
    order = [0,1,2,3]
else:
    order = [2,0,3,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False,fontsize=11.5,handletextpad=0.3,handlelength=1.6,borderaxespad=0.2)

ax2 = fig0.add_axes([0.7, 0.24, 0.255, 0.29])
ax2.plot(datasave[:,0],datasave[:,1]*gmol,ls='-',c=c2[1],zorder=0,lw=1,label='full PPMS data')
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='minor', labelsize=10)
ax2.set_xlim([0,390])
if savetag:
    plt.savefig(os.path.join(direc,'molar_heat_capacity_co.png'),dpi=600,transparent=True)


rho = np.interp(data[:,0],T,4*gmol/1e3/V/ag/1e-30)
idx8 = np.where(df[df.columns[8]]>392)[0][0]
x8 = df[df.columns[8]][:idx8]
y8 = df[df.columns[9]][:idx8]
data0 = np.zeros((len(x8),2))
data0[:,0] = x8
data0[:,1] = y8/gmol

idx1 = np.where(data0[:,0] < Ttr)[0]
idx2 = np.where(data0[:,0] > Ttr)[0]
idx_temp1 = np.where(data0[:,0] > 286)[0][0]
print('td',resi2.x[1])
print(data0[0:idx_temp1,0])

idx_temp2 = np.where(data0[:,0] > 371)[0][0]
resi_scale1 = least_squares(resiL_scale,1,args=(data0[:idx_temp1,0],data0[:idx_temp1,1]*gmol,resi2.x[0],resi2.x[1],resi2.x[2],resi2.x[3]),bounds=(0.2,2))
resi_scale2 = least_squares(resiL_scale,1,args=(data0[idx_temp2:,0],data0[idx_temp2:,1]*gmol,resi3.x[0],resi3.x[1],resi3.x[2],resi3.x[3]),bounds=(0.2,2))

figC = plt.figure(figsize=(5,4.4))
iskip = 65
iskip2 = 71
mss = 24
plt.scatter(data[iskip2:,0],data[iskip2:,1]*rho[iskip2:]/gmol*1000/1e6,mss,edgecolors=c2[1],zorder=10,alpha=0.9,facecolors=c2[0])
plt.scatter(data[iskip:iskip2:2,0],data[iskip:iskip2:2,1]*rho[iskip:iskip2:2]/gmol*1000/1e6,mss,edgecolors=c2[1],zorder=10,alpha=0.9,facecolors=c2[0])
plt.scatter(data[:iskip:3,0],data[:iskip:3,1]*rho[:iskip:3]/gmol*1000/1e6,mss,edgecolors=c2[1],zorder=10,alpha=0.9,facecolors=c2[0],label=r'Fe[HB(tz)$\mathrm{_3}$]$\mathrm{_2}$, PPMS')

yc1 = cdebye_linear(data[np.where(data[:,0]<Ttr)[0],0],resi2.x[0],resi2.x[1],resi2.x[2],resi2.x[3])*np.interp(data[np.where(data[:,0]<Ttr)[0],0],data[:,0],rho)/gmol*1000
yc2 = cdebye_linear(data[np.where(data[:,0]>=Ttr)[0],0],resi3.x[0],resi3.x[1],resi3.x[2],resi3.x[3])*np.interp(data[np.where(data[:,0]>=Ttr)[0],0],data[:,0],rho)/gmol*1000 
csdebye = np.concatenate((yc1,yc2))
csdebye = np.array([data[np.where((data[:,0]<Ttr) | (data[:,0] >=Ttr))[0],0],csdebye]).reshape(2,-1).T

print(idxb)
x = data[idxb,0]
y = data[idxb,1] * rho[idxb]/gmol*1000
X_Y_Spline = make_interp_spline(x, y)
plt.subplots_adjust(left=0.175, right=0.95, top=0.94, bottom=0.17)
print('SPLINE:',np.interp([250,300,350,375,400],data[:,0],X_Y_Spline(data[:,0])/1e6))
T1_temp = data[np.where(data[:,0]<=Ttr1)[0],0]
CT1_temp = cdebye_linear(T1_temp,resi2.x[0],resi2.x[1],resi2.x[2],resi2.x[3])*np.interp(T1_temp,data[:,0],rho)/gmol*1000/1e6
CT1 = np.interp(T1,T1_temp,CT1_temp)
idx = np.argwhere(T1 > Ttr1)
CT1[idx] = (CT1_temp[-1]-CT1_temp[-2])/(T1_temp[-1]-T1_temp[-2])*(T1[idx]-T1_temp[-2])+CT1_temp[-2]
plt.plot(T1,CT1,ls='-',c='C0',zorder=3,lw=2.5)
print('=================================\n')
print('Debye:',np.interp([250,300,350,375,400],T1,cdebye_linear(T1,resi2.x[0],resi2.x[1],resi2.x[2],resi2.x[3])*np.interp(T1,data[:,0],rho)/gmol*1000/1e6))

T2_temp = data[np.where(data[:,0]>=Ttr2)[0],0]
CT2_temp = cdebye_linear(T2_temp,resi3.x[0],resi3.x[1],resi3.x[2],resi3.x[3])*np.interp(T2_temp,data[:,0],rho)/gmol*1000/1e6
CT2 = np.interp(T2,T2_temp,CT2_temp)
idx = np.argwhere(T2 < Ttr2)
CT2[idx] = (CT2_temp[1]-CT2_temp[0])/(T2_temp[1]-T2_temp[0])*(T2[idx]-T2_temp[0])+CT2_temp[0]
plt.plot(T2,CT2,ls='-',c='C0',zorder=3,lw=2.5,label='Debye-linear model, PPMS')
cccc = sns.light_palette('C1',5)
Dxdsc = data0[:,0]
Dydsc = data0[:,1]*np.interp(data0[:,0],T,4*gmol/1e3/V/ag/1e-30)/1000
cdsc = sns.light_palette(cccc[-2],5)
intvd = 150
plt.scatter(Dxdsc[::intvd],Dydsc[::intvd],mss,edgecolors=cccc[-2],facecolors=cdsc[1],label=r'Fe[HB(tz)$\mathrm{_3}$]$\mathrm{_2}$, DSC',marker='D',zorder=302,alpha=0.8)
idx_temp = np.where(T1>data0[0,0])[0]
plt.plot(T1,CT1*resi_scale1.x,ls='--',c='C1',zorder=300,lw=2.5,label='Debye-linear model, DSC')
Tcomb = np.concatenate((T1[np.where(T1>Dxdsc[0])],T2))
tdsc = np.concatenate((data0[:idx_temp1,0],data0[idx_temp2:,0]))
ddsc = np.concatenate((data0[:idx_temp1,1]*gmol,data0[idx_temp2:,1]*gmol)) *np.interp(tdsc,data[:,0],rho)/gmol*1000/1e6
DSC_X_Y_Spline = CubicSpline(tdsc,ddsc)

idx_temp = np.where(T2<data0[-1,0])[0]
plt.plot(T2,CT2*resi_scale2.x,ls='--',c='C1',zorder=300,lw=2.5)
plt.plot(Tcomb,DSC_X_Y_Spline(Tcomb),ls=':',zorder=301,c='grey',label='spline baseline, DSC',lw=1.5)
cdscspline = np.array([Tcomb,DSC_X_Y_Spline(Tcomb)]).reshape(2,-1).T
df_dsc = pd.DataFrame(cdscspline)
print('=================================\n')
print('DSC (Spline):')
print(df_dsc.to_string(header=False, index=False))
print('=================================\n')
csdebye = np.concatenate((CT1,CT2))
csdebye = np.array([np.concatenate((T1,T2)),csdebye*1e6]).reshape(2,-1).T
df = pd.DataFrame(csdebye)
print('Debye (finalised):')
print(df.to_string(header=False, index=False))
print('=================================\n')
print((CT2[0]-CT1[-1])/CT1[-1]*100,'percent, volume')

plt.scatter(data[[iend-1,istart],0],data[[iend-1,istart],1]*rho[[iend-1,istart]]/1e3/gmol,230,edgecolors=None,facecolors='gray',marker='|',zorder=0)
handles, labels = plt.gca().get_legend_handles_labels()
if os.name == 'nt':
    order = [0,1,2,3,4]
else:
    order = [3,0,4,1,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False,fontsize=11.5,handletextpad=0.3,handlelength=1.6,borderaxespad=0.2)

print('==',data[[iend,istart],1]*rho[[iend,istart]]/1e3/gmol)
plt.xlim([0,390])
plt.ylim([0,2.5])
plt.xlabel('Temperature (K)',labelpad=6)
plt.ylabel(r'Heat capacity ($\times\mathregular{10^6 J\,m^{-3}\,K^{-1}}$)',labelpad=6)
axc = figC.add_axes([0.70, 0.24, 0.233, 0.26])
axc.plot(Dxdsc[::],Dydsc[::],ls='--',c='C1',zorder=1,lw=1,label='full PPMS data')
axc.plot(data[::,0],data[::,1]*rho[::]/gmol*1000/1e6,c=c2[1],zorder=0)
axc.set_xlim([0,390])
axc.tick_params(axis='both', which='major', labelsize=10)
axc.tick_params(axis='both', which='minor', labelsize=10)
if savetag:
    plt.savefig(os.path.join(direc,'Cp_PPMS_linear.png'),transparent=True,dpi=600)

csdebye_scale = np.concatenate((CT1*resi_scale1.x,CT2*resi_scale2.x))
csdebye_scale = np.array([np.concatenate((T1,T2)),csdebye_scale*1e6]).reshape(2,-1).T
df1 = pd.DataFrame(csdebye_scale)
print('DSC Debye (finalised):')
print(df1.to_string(header=False, index=False))
print('=================================\n')
ybl = X_Y_Spline(data[:,0]) 
cspline = np.array([data[:,0],ybl]).reshape(2,-1).T
df = pd.DataFrame(cspline)
print('Spline finalized (PPMS):')
print(df.to_string(header=False, index=False))
print('dS_spline_vol',np.trapz(data[:,1]*rho/gmol*1000-X_Y_Spline(data[:,0]))/330.25/gmol*1000)
plt.show()