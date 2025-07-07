import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib

# plot style
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='6'
plt.rcParams.update({'font.size': 15})
font = {'family' : 'Arial'}
matplotlib.rc('font', **font)

savetag = 1
direc = 'data'

def getTemp(fname):
    ff = fname.split('_')
    for i in range(len(ff)):
        if len(ff[i].split('C'))==2:
            ff = ff[i].split('C')
            break
    return float(ff[0])

def getT_idx(T, Tlist):
    ir = -1
    for i in range(len(Tlist)):
        if abs(T-Tlist[i]) < 1.5:
            ir = i
            break
    return ir


def return_data(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    nl = len(lines)
    l0 = lines[0].split()
    rad = float(l0[-6])
    power = float(l0[-2])
    data = np.zeros((nl-2, 3))
    for i in range(2, nl):
        data[i-2, :] = np.array(lines[i].split())
    return rad, power, data

def return_data_csv(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    nl = len(lines)
    data = np.zeros((nl, 2))
    for i in range(nl):
        data[i, :] = np.array(lines[i].split(','))
    return data

def write_avg_data(data,name,header):
    f = open(name,'w')
    f.write(header)
    f.write('Frequency (Hz)\tR (Vpp)\tTheta (degree)\n')
    for i in range(data.shape[0]):
        f.write('{:.7e}\t{:.7e}\t{:.7e}'.format(data[i,0],data[i,1],data[i,2])+'\n')
    f.close()

def return_header(fname):
    f = open(fname,'r')
    lines = f.readlines()
    f.close()
    l0 = lines[0].split()
    return lines[0]
   
cvtype = 'SCO_DSC_spline' 
cc = sns.color_palette()
x2 = loadmat(os.path.join(direc,'best_fits_'+cvtype+'.mat'))
idx0 = np.arange(13)
tfinal2 = x2['tfinal'][0][idx0]
kfinal2 = x2['kfinal'][0][idx0]
dkfinal2 = x2['dkfinal'][0][idx0]

cvtype = 'SCO_DSC_linear' 
cc = sns.color_palette()
x = loadmat(os.path.join(direc,'best_fits_'+cvtype+'.mat'))
idx0 = np.arange(13)
tfinal = x['tfinal'][0][idx0]
kfinal = x['kfinal'][0][idx0]
dkfinal = x['dkfinal'][0][idx0]

cvtype1 = 'SCO_PPMS_linear' 
x1 = loadmat(os.path.join(direc,'best_fits_'+cvtype1+'.mat'))
idx1 = np.arange(13)
tfinal1 = x1['tfinal'][0][idx1]
kfinal1 = x1['kfinal'][0][idx1]
dkfinal1 = x1['dkfinal'][0][idx1]
apap = 0.98
fg, ax = plt.subplots(figsize=(5.0,4.4))

shading = 0
ec = '0.6'
elw = 1.5
cccc = sns.light_palette('C0',3)
cccc1 = sns.light_palette('C1',3)
cccc2 = sns.light_palette('C7',3)
if shading:
    ax.scatter(tfinal1[:],kfinal1[:],80,facecolors=cc[0],edgecolors='k',linewidth=1.2,alpha=apap,zorder=3)
    plt.fill_between(x1['tfinal'][0][idx0], x1['kfinal'][0][idx0]-x1['dkfinal'][0][idx0],x1['kfinal'][0][idx0]+x1['dkfinal'][0][idx0],color='gray',alpha=0.1)
else:
    Sid = 6
    ax.errorbar(tfinal1[:Sid],kfinal1[:Sid],dkfinal1[:Sid],fmt='o',ms=9,capsize=4,markerfacecolor=cccc[1],markeredgecolor=cc[0],markeredgewidth=1.2,alpha=apap,zorder=4,ecolor=cc[0],elinewidth=elw)
    ax.errorbar(tfinal1[Sid:],kfinal1[Sid:],dkfinal1[Sid:],fmt='o',ms=9,capsize=4,markerfacecolor=cccc[1],markeredgecolor=cc[0],markeredgewidth=1.2,alpha=apap,zorder=4,ecolor=cc[0],elinewidth=elw)
    ax.errorbar(tfinal[:Sid],kfinal[:Sid],dkfinal[:Sid],fmt='D',ms=9,capsize=4,markerfacecolor=cccc1[1],markeredgecolor=cc[1],markeredgewidth=1.2,alpha=apap,zorder=3,ecolor=cc[1],elinewidth=elw)
    ax.errorbar(tfinal[Sid:],kfinal[Sid:],dkfinal[Sid:],fmt='D',ms=9,capsize=4,markerfacecolor=cccc1[1],markeredgecolor=cc[1],markeredgewidth=1.2,alpha=apap,zorder=3,ecolor=cc[1],elinewidth=elw)
    ax.errorbar(tfinal2[:Sid],kfinal2[:Sid],dkfinal2[:Sid],fmt='D',ms=9,capsize=4,markerfacecolor='none',markeredgecolor=cc[7],markeredgewidth=1.2,alpha=apap,zorder=3,ecolor=cc[7],elinewidth=elw)
    ax.errorbar(tfinal2[Sid:],kfinal2[Sid:],dkfinal2[Sid:],fmt='D',ms=9,capsize=4,markerfacecolor='none',markeredgecolor=cc[7],markeredgewidth=1.2,alpha=apap,zorder=3,ecolor=cc[7],elinewidth=elw)

print(kfinal1)
print(kfinal1[5],kfinal1[8],kfinal1[5]/kfinal1[8])
dk1 = dkfinal1[5]
k1 = kfinal1[5]
dk2 = dkfinal1[8]
k2 = kfinal1[8]
print('error',np.sqrt((k1*dk2/k2**2)**2+(dk1/k2)**2))

idx = [6,7]
tfinal = x['tfinal'][0][idx]
kfinal = x['kfinal'][0][idx]
dkfinal = x['dkfinal'][0][idx]
idx1 = [6,7]
tfinal1 = x1['tfinal'][0][idx1]
kfinal1 = x1['kfinal'][0][idx1]
dkfinal1 = x1['dkfinal'][0][idx1]
idx2 = [6,7]
tfinal2 = x2['tfinal'][0][idx2]
kfinal2 = x2['kfinal'][0][idx2]
dkfinal2 = x2['dkfinal'][0][idx2]

if shading:
    ax.scatter(tfinal1[:],kfinal1[:],80,facecolors='white',edgecolors='k',linewidth=1.2,alpha=apap,zorder=3)
else:
    ax.errorbar(tfinal1,kfinal1,dkfinal1,fmt='o',capsize=4,ms=9,markerfacecolor=cccc[1],markeredgecolor=cc[0],elinewidth=elw,markeredgewidth=1.2,alpha=apap,zorder=6,ecolor=cc[0],label=r'$C_p$ from PPMS')
    ax.errorbar(tfinal,kfinal,dkfinal,fmt='D',capsize=4,ms=9,markerfacecolor=cccc1[1],markeredgecolor=cc[1],elinewidth=elw,markeredgewidth=1.2,alpha=apap,zorder=5,ecolor=cc[1],label=r'$C_p$ from DSC')
    ax.errorbar(tfinal2,kfinal2,dkfinal2,fmt='D',capsize=4,ms=9,markerfacecolor='none',markeredgecolor=cc[7],elinewidth=elw,markeredgewidth=1.2,alpha=apap,zorder=5,ecolor=cc[7],label=r'$C_p$ from DSC (spline)')

ax.set_ylim([0.0,1.25])
fg.legend(frameon=False,handletextpad=0.025,borderaxespad=0.02,borderpad=0.2,bbox_to_anchor=(0.97, 0.88),fontsize=11)
n= 256 
    
# get the list of color from colormap
cmap = plt.get_cmap('Greys_r')
colors_r = cmap(np.linspace(0, 1, n))    # take the standard colormap # 'right-part'
colors_l = colors_r[::-1]                # take the first list of color and flip the order # "left-part"

# combine them and build a new colormap
colors = np.vstack((colors_l, colors_r))

polygon1 = ax.fill_between([326.6,336.6], [-0.1,-0.1],[1.65,1.65],color='gray',alpha=0.01,zorder=0)
polygon = ax.fill_between([326.6,336.6], [-0.1,-0.1],[1.65,1.65],color='none',alpha=0.0,zorder=1)

xlim = plt.xlim()
ylim = plt.ylim()
verts = np.vstack([p.vertices for p in polygon.get_paths()])
mymap = mcolors.LinearSegmentedColormap.from_list('none', colors)
gradient = plt.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=mymap, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()],alpha=0.2,zorder=0)
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

plt.xlabel('Temperature (K)',labelpad=6)
plt.ylabel(r'Thermal conductivity ($\mathregular{W\,{m^{-1}} {K^{-1}}}$)',labelpad=4)
plt.xlim([295,365])
plt.subplots_adjust(left=0.195, right=0.97, top=0.92, bottom=0.15)

if savetag:
    plt.savefig('kappa_ppms_linear_DSC.png',transparent=True,dpi=600)
idx = np.arange(13)
tfinal = x['tfinal'][0][idx]
kfinal = x['kfinal'][0][idx]
dkfinal = x['dkfinal'][0][idx]
fg, ax1 = plt.subplots(figsize=(5.0,4.4))

cv_debye="""
1.780988 -9.608499e+03
  3.438370 -2.160812e+03
  5.095752  5.409620e+03
  6.753134  1.316201e+04
  8.410516  2.115769e+04
 10.067898  2.945190e+04
 11.725280  3.810998e+04
 13.382662  4.717564e+04
 15.040044  5.670406e+04
 16.697426  6.672175e+04
 18.354808  7.722277e+04
 20.012190  8.819903e+04
 21.669572  9.960259e+04
 23.326954  1.113957e+05
 24.984336  1.234808e+05
 26.641718  1.357976e+05
 28.299099  1.482805e+05
 29.956481  1.608626e+05
 31.613863  1.734817e+05
 33.271245  1.860836e+05
 34.928627  1.986262e+05
 36.586009  2.110800e+05
 38.243391  2.234281e+05
 39.900773  2.356168e+05
 41.558155  2.476318e+05
 43.215537  2.595084e+05
 44.872919  2.712216e+05
 46.530301  2.827014e+05
 48.187683  2.940668e+05
 49.845065  3.052123e+05
 51.502447  3.161927e+05
 53.159829  3.270388e+05
 54.817211  3.376600e+05
 56.474593  3.482066e+05
 58.131975  3.585019e+05
 59.789357  3.687554e+05
 61.446739  3.787667e+05
 63.104121  3.887373e+05
 64.761503  3.985090e+05
 66.418885  4.082104e+05
 68.076267  4.177855e+05
 69.733649  4.272339e+05
 71.391031  4.366521e+05
 73.048413  4.458655e+05
 74.705795  4.550788e+05
 76.363177  4.641582e+05
 78.020559  4.731552e+05
 79.677941  4.821522e+05
 81.335323  4.909597e+05
 82.992704  4.997593e+05
 84.650086  5.085336e+05
 86.307468  5.171546e+05
 87.964850  5.257756e+05
 89.622232  5.343715e+05
 91.279614  5.428322e+05
 92.936996  5.512930e+05
 94.594378  5.597537e+05
 96.251760  5.680711e+05
 97.909142  5.763884e+05
 99.566524  5.847057e+05
101.223906  5.928762e+05
102.881288  6.009752e+05
104.538670  6.090742e+05
106.196052  6.171732e+05
107.853434  6.252722e+05
109.510816  6.333713e+05
111.168198  6.413618e+05
112.825580  6.492882e+05
114.482962  6.572147e+05
116.140344  6.651411e+05
117.797726  6.730676e+05
119.455108  6.809940e+05
121.112490  6.888398e+05
122.769872  6.966356e+05
124.427254  7.044314e+05
126.084636  7.122272e+05
127.742018  7.200230e+05
129.399400  7.278188e+05
131.056782  7.355576e+05
132.714164  7.432523e+05
134.371546  7.509471e+05
136.028928  7.586418e+05
137.686310  7.663365e+05
139.343691  7.740312e+05
141.001073  7.816853e+05
142.658455  7.893005e+05
144.315837  7.969158e+05
145.973219  8.045311e+05
147.630601  8.121463e+05
149.287983  8.197616e+05
150.945365  8.273460e+05
152.602747  8.348940e+05
154.260129  8.424420e+05
155.917511  8.499900e+05
157.574893  8.575381e+05
159.232275  8.650861e+05
160.889657  8.726127e+05
162.547039  8.801087e+05
164.204421  8.876047e+05
165.861803  8.951006e+05
167.519185  9.025966e+05
169.176567  9.100926e+05
170.833949  9.175723e+05
172.491331  9.250252e+05
174.148713  9.324781e+05
175.806095  9.399309e+05
177.463477  9.473838e+05
179.120859  9.548366e+05
180.778241  9.622771e+05
182.435623  9.696936e+05
184.093005  9.771101e+05
185.750387  9.845266e+05
187.407769  9.919430e+05
189.065151  9.993595e+05
190.722533  1.006767e+06
192.379915  1.014154e+06
194.037296  1.021540e+06
195.694678  1.028927e+06
197.352060  1.036313e+06
199.009442  1.043700e+06
200.666824  1.051093e+06
202.324206  1.058503e+06
203.981588  1.065913e+06
205.638970  1.073324e+06
207.296352  1.080734e+06
208.953734  1.088145e+06
210.611116  1.095551e+06
212.268498  1.102942e+06
213.925880  1.110333e+06
215.583262  1.117724e+06
217.240644  1.125115e+06
218.898026  1.132505e+06
220.555408  1.139894e+06
222.212790  1.147268e+06
223.870172  1.154642e+06
225.527554  1.162016e+06
227.184936  1.169390e+06
228.842318  1.176763e+06
230.499700  1.184136e+06
232.157082  1.191495e+06
233.814464  1.198854e+06
235.471846  1.206213e+06
237.129228  1.213572e+06
238.786610  1.220931e+06
240.443992  1.228290e+06
242.101374  1.235629e+06
243.758756  1.242967e+06
245.416138  1.250306e+06
247.073520  1.257644e+06
248.730901  1.264983e+06
250.388283  1.272321e+06
252.045665  1.279469e+06
253.703047  1.286615e+06
255.360429  1.293761e+06
257.017811  1.300907e+06
258.675193  1.308053e+06
260.332575  1.315200e+06
261.989957  1.322323e+06
263.647339  1.329446e+06
265.304721  1.336569e+06
266.962103  1.343692e+06
268.619485  1.350814e+06
270.276867  1.357937e+06
271.934249  1.365039e+06
273.591631  1.372140e+06
275.249013  1.379240e+06
276.906395  1.386341e+06
278.563777  1.393441e+06
280.221159  1.400542e+06
281.878541  1.407622e+06
283.535923  1.414701e+06
285.193305  1.421780e+06
286.850687  1.428859e+06
288.508069  1.435938e+06
290.165451  1.443017e+06
291.822833  1.450068e+06
293.480215  1.457115e+06
295.137597  1.464162e+06
296.794979  1.471209e+06
298.452361  1.478257e+06
300.109743  1.485304e+06
301.767125  1.492006e+06
303.424507  1.498664e+06
305.081888  1.505322e+06
306.739270  1.511980e+06
308.396652  1.518638e+06
310.054034  1.525296e+06
311.711416  1.531929e+06
313.368798  1.538555e+06
315.026180  1.545181e+06
316.683562  1.551793e+06
318.340944  1.558398e+06
319.998326  1.565004e+06
321.655708  1.571598e+06
323.313090  1.578185e+06
324.970472  1.584494e+06
326.627854  1.590250e+06
328.285236  1.595700e+06
329.942618  1.601150e+06
331.600000  1.606599e+06
331.600100  1.852886e+06
331.866525  1.853077e+06
332.132951  1.853268e+06
332.399376  1.853459e+06
332.665801  1.853650e+06
332.932226  1.853841e+06
333.198652  1.854032e+06
333.465077  1.854223e+06
333.731502  1.854414e+06
333.997927  1.854605e+06
334.264353  1.854796e+06
334.530778  1.854987e+06
334.797203  1.855178e+06
335.063629  1.855369e+06
335.330054  1.855560e+06
335.596479  1.855751e+06
335.862904  1.855941e+06
336.129330  1.856200e+06
336.395755  1.856323e+06
336.662180  1.856514e+06
336.928606  1.856705e+06
337.195031  1.856896e+06
337.461456  1.857087e+06
337.727881  1.857278e+06
337.994307  1.857469e+06
338.260732  1.857662e+06
338.527157  1.857865e+06
338.793582  1.858069e+06
339.060008  1.858272e+06
339.326433  1.858475e+06
339.592858  1.858679e+06
339.859284  1.858882e+06
340.125709  1.859085e+06
340.392134  1.859366e+06
340.658559  1.859683e+06
340.924985  1.860000e+06
341.191410  1.860317e+06
341.457835  1.860634e+06
341.724261  1.860951e+06
341.990686  1.861268e+06
342.257111  1.861585e+06
342.523536  1.861902e+06
342.789962  1.862219e+06
343.056387  1.862536e+06
343.322812  1.862853e+06
343.589237  1.863170e+06
343.855663  1.863487e+06
344.122088  1.863804e+06
344.388513  1.864121e+06
344.654939  1.864438e+06
344.921364  1.864755e+06
345.187789  1.865072e+06
345.454214  1.865390e+06
345.720640  1.865708e+06
345.987065  1.866026e+06
346.253490  1.866344e+06
346.519915  1.866662e+06
346.786341  1.866980e+06
347.052766  1.867298e+06
347.319191  1.867617e+06
347.585617  1.867935e+06
347.852042  1.868253e+06
348.118467  1.868571e+06
348.384892  1.868889e+06
348.651318  1.869207e+06
348.917743  1.869525e+06
349.184168  1.869843e+06
349.450594  1.870161e+06
349.717019  1.870479e+06
349.983444  1.870797e+06
350.249869  1.871174e+06
350.516295  1.871626e+06
350.782720  1.872078e+06
351.049145  1.872530e+06
351.315570  1.872982e+06
351.581996  1.873434e+06
351.848421  1.873887e+06
352.114846  1.874339e+06
352.381272  1.874791e+06
352.647697  1.875243e+06
352.914122  1.875695e+06
353.180547  1.876147e+06
353.446973  1.876599e+06
353.713398  1.877051e+06
353.979823  1.877504e+06
354.246248  1.877956e+06
354.512674  1.878408e+06
354.779099  1.878860e+06
355.045524  1.879312e+06
355.311950  1.879764e+06
355.578375  1.880216e+06
355.844800  1.880669e+06
356.111225  1.881121e+06
356.377651  1.881573e+06
356.644076  1.882025e+06
356.910501  1.882477e+06
357.176927  1.882929e+06
357.443352  1.883381e+06
357.709777  1.883833e+06
357.976202  1.884286e+06
358.242628  1.884738e+06
358.509053  1.885190e+06
358.775478  1.885642e+06
359.041903  1.886094e+06
359.308329  1.886546e+06
359.574754  1.886998e+06
359.841179  1.887450e+06
360.107605  1.887903e+06
360.374030  1.888403e+06
360.640455  1.888908e+06
360.906880  1.889414e+06
361.173306  1.889919e+06
361.439731  1.890424e+06
361.706156  1.890929e+06
361.972582  1.891434e+06
362.239007  1.891940e+06
362.505432  1.892445e+06
362.771857  1.892950e+06
363.038283  1.893455e+06
363.304708  1.893960e+06
363.571133  1.894466e+06
363.837558  1.894971e+06
364.103984  1.895476e+06
364.370409  1.895981e+06
364.636834  1.896486e+06
364.903260  1.896991e+06
365.169685  1.897497e+06
365.436110  1.898002e+06
365.702535  1.898507e+06
365.968961  1.899012e+06
366.235386  1.899517e+06
366.501811  1.900023e+06
366.768236  1.900528e+06
367.034662  1.901033e+06
367.301087  1.901538e+06
367.567512  1.902043e+06
367.833938  1.902549e+06
368.100363  1.903054e+06
368.366788  1.903559e+06
368.633213  1.904064e+06
368.899639  1.904569e+06
369.166064  1.905075e+06
369.432489  1.905580e+06
369.698915  1.906085e+06
369.965340  1.906590e+06
370.231765  1.907095e+06
370.498190  1.907601e+06
370.764616  1.908106e+06
371.031041  1.908611e+06
371.297466  1.909116e+06
371.563891  1.909621e+06
371.830317  1.910127e+06
372.096742  1.910632e+06
372.363167  1.911137e+06
372.629593  1.911642e+06
372.896018  1.912147e+06
373.162443  1.912652e+06
373.428868  1.913158e+06
373.695294  1.913663e+06
373.961719  1.914168e+06
374.228144  1.914673e+06
374.494569  1.915178e+06
374.760995  1.915684e+06
375.027420  1.916189e+06
375.293845  1.916694e+06
375.560271  1.917199e+06
375.826696  1.917704e+06
376.093121  1.918210e+06
376.359546  1.918715e+06
376.625972  1.919220e+06
376.892397  1.919725e+06
377.158822  1.920230e+06
377.425248  1.920736e+06
377.691673  1.921241e+06
377.958098  1.921746e+06
378.224523  1.922251e+06
378.490949  1.922756e+06
378.757374  1.923262e+06
379.023799  1.923767e+06
379.290224  1.924272e+06
379.556650  1.924777e+06
379.823075  1.925282e+06
380.089500  1.925787e+06
380.355926  1.926293e+06
380.622351  1.926798e+06
380.888776  1.927303e+06
381.155201  1.927808e+06
381.421627  1.928313e+06
381.688052  1.928819e+06
381.954477  1.929324e+06
382.220903  1.929829e+06
382.487328  1.930334e+06
382.753753  1.930839e+06
383.020178  1.931345e+06
383.286604  1.931850e+06
383.553029  1.932355e+06
383.819454  1.932860e+06
384.085879  1.933365e+06
384.352305  1.933871e+06
384.618730  1.934376e+06
"""
from io import StringIO
cv_debye= np.genfromtxt(StringIO(cv_debye),dtype=float)
cv = np.interp(tfinal, cv_debye[:,0], cv_debye[:,1])
ip1 = [0,1,2,3,4,5]
ip2 = [8,9,10,11,12] 
ip3 = [6,7]
plt.errorbar(tfinal[ip1],kfinal[ip1]/cv[ip1]/1e-6,dkfinal[ip1]/cv[ip1]/1e-6,fmt='o',capsize=0,ms=9,markerfacecolor=cc[0],markeredgecolor='k',ecolor=ec,markeredgewidth=1.2,alpha=apap,elinewidth=elw)
plt.errorbar(tfinal[ip2],kfinal[ip2]/cv[ip2]/1e-6,dkfinal[ip2]/cv[ip2]/1e-6,fmt='o',capsize=0,ms=9,markerfacecolor=cc[3],markeredgecolor='k',ecolor=ec,markeredgewidth=1.2,alpha=apap,elinewidth=elw)
if shading and len(ip3) > 0:
    ax1.scatter(tfinal[ip3],kfinal[ip3]/cv[ip3]/1e-6,80,facecolors='white',edgecolors='k',linewidth=1.2,alpha=apap,zorder=3)
elif len(ip3) > 0:
    ax1.errorbar(tfinal[ip3],kfinal[ip3]/cv[ip3]/1e-6,dkfinal[ip3]/cv[ip3]/1e-6,fmt='o',capsize=0,ms=9,markerfacecolor='white',markeredgecolor='k',elinewidth=elw,markeredgewidth=1.2,alpha=apap,zorder=4,ecolor=ec)
polygon1 = ax1.fill_between([326,336], [0,0],[1.65,1.65],color='gray',alpha=0.01,zorder=0)
polygon = ax1.fill_between([326,336], [0,0],[1.1,1.1],color='none',alpha=0.0,zorder=1)
aa1 = (kfinal[ip1[-1]]/cv[ip1[-1]])
aa2 = (kfinal[ip2[0]]/cv[ip2[0]])
daa1 = dkfinal[ip1[-1]]/cv[ip1[-1]]
daa2 = dkfinal[ip2[0]]/cv[ip2[0]]
deltaa = np.sqrt((aa1*daa2/aa2**2)**2+(daa1/aa2)**2)


plt.text(333.5,0.5,r'$\alpha_\mathrm{LS}/\alpha_\mathrm{HS}$ = ' +str(round(aa1/aa2,1))+r' $\pm$ '+str(round(deltaa,1)),fontsize=14)
xlim = plt.xlim()
ylim = plt.ylim()
verts = np.vstack([p.vertices for p in polygon.get_paths()])
mymap = mcolors.LinearSegmentedColormap.from_list('none', colors)
gradient = plt.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=mymap, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()],alpha=0.2,zorder=0)
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

plt.xlim([295,365])
plt.ylim([0,0.85])
#
plt.xlabel('Temperature (K)')
plt.ylabel(r'Thermal diffusivity ($\mathregular{{mm^2} {s^{-1}}}$)',labelpad=6)
plt.subplots_adjust(left=0.175, right=0.95, top=0.92, bottom=0.15)
plt.show()

