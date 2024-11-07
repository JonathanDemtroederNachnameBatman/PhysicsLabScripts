import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from scipy import optimize
from scipy import signal
from scipy import stats
from fractions import Fraction
from lib import helpers as hp

def einstein():
    NA = 1
    kB = 1
    h = 1
    nuE = 1
    NAkB = 8.31446261815
    x = np.linspace(0.001, 1, 200)
    C = 3*NAkB* (1/x**2) * np.exp(1/x) / (np.exp(1/x)-1)**2
    
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(label='$T/\Theta_E$'), hp.AxisConfig(label='$C_{mol,V}$'))
    ax.plot(x, C)
    plt.show()
    
def f(L, x):
    return x*L.slope+L.intercept

def ef(L, x):
    return np.sqrt()

DT = np.array([4.42908571, 4.88768431, 4.59450002, 4.39102696, 3.96825058, 3.88863478, 4.24635268, 3.55090576, 4.45079413])
eDT = np.array([0.07537202, 0.11302257, 0.15681601, 0.2259476, 0.30145445, 0.34877468, 0.29438397, 0.30139377, 0.38141431]) 
TP = np.array([108.85916667, 116.767, 145.0615, 172.23316667, 218.77788333, 254.04736667, 277.65233167, 303.82515, 337.32826667])
eTP = np.array([0.30023486, 0.33904424, 0.31491705, 0.29254909, 0.30044745, 0.33113839, 0.38926939, 0.43181581, 0.39690989])
C = np.array([11.21149909, 11.91256048, 16.64272392, 22.64926994, 26.13086566, 29.59469801, 31.30631475, 37.63998682, 33.42714298])
eC = np.array([0.25032149, 0.32186988, 0.61037072, 1.20261207, 2.01418781, 2.68200686, 2.20752622, 3.23139976, 2.89650026])

def plots():    
    fig, ax = plt.subplots(figsize=[16/2.54, 8/2.54])
    plt.gcf().subplots_adjust(top=0.975, bottom=0.175)
    j1 = np.array([200, 140, 140, 240, 150, 140, 200, 160, 170])
    j2 = np.array([400, 250, 200, 190, 180, 190, 250, 280, 240])
    k = np.array([385, 300, 294, 407, 353, 332, 399, 363, 396])
    for i in range(1, 10):
        ax.config(hp.AxisConfig(label='t in s', majors=30 if i >= 2 else 60, minors=4), hp.AxisConfig(label='T in K', majors=1, minors=2))
        D = pd.read_csv(f'data/v29/{i}.txt', decimal=',', header=None, dtype=np.float64, skiprows=8, sep='\t').to_numpy().T
        t = D[0]
        T = D[1] + 273.15
        # plot data
        ax.plot(t[:j1[i-1]], T[:j1[i-1]], '.', markersize=1)
        ax.plot(t[j1[i-1]:-j2[i-1]], T[j1[i-1]:-j2[i-1]], '.', markersize=1)
        ax.plot(t[-j2[i-1]:], T[-j2[i-1]:], '.', markersize=1)
        # lin fit für anfang und ende
        L1 = stats.linregress(t[:j1[i-1]], T[:j1[i-1]])
        L2 = stats.linregress(t[-j2[i-1]:], T[-j2[i-1]:])
        length = 600 if i > 2 else 800
        # plot fit
        ax.plot(t[:length], f(L1, t[:length]), '--')
        ax.plot(t[-length:], f(L2, t[-length:]), '--')
        # plot vertical
        j = k[i-1]
        ax.plot([t[j], t[j]], [f(L1, t[j]), f(L2, t[j])], 'k--', linewidth=1)
        # plot horizontal            
        s = np.average(T[j-30:j+30])
        es = np.std(T[j-30:j+30])
        #TP[i-1] = s
        #eTP[i-1] = es
        ax.plot([t[0], t[-1]], [s, s], 'k--', linewidth=1)
        #ax.text_box(f'Index: {i}', 0.025, 0.975)
        #plt.show()
        dT = f(L2, t[j]) - f(L1, t[j])
        #eDT[i-1] = np.sqrt((t[j]*L1.stderr)**2 + (t[j]*L2.stderr)**2 + L1.intercept_stderr**2 + L2.intercept_stderr**2 + ((L2.slope-L1.slope)*1)**2)
        #print(f'{i}: DT={round(dT, 3)}')
        #DT[i-1] = dT
        #ax.text_box(f'$\Delta T={round(dT, 2)}\pm{round(eDT[i-1], 2)} K$\n$T_P={round(s, 2)}\pm{round(es, 2)}$', 0.975, 0.45, horizontal='right')
        ax.text_box('$\Delta T={:5.2f}\pm{:3.2f}$\n$T_P={:5.2f}\pm{:3.2f}$'.format(dT, eDT[i-1], s, es), 0.975, 0.45, horizontal='right')
        plt.savefig(f'tmp/v29_{i}.png')
        ax.cla()
        if False:
            w1 = 0; w2 = 0
            for l in range(j): w1 += (T[l] - f(L1, t[l])) * np.abs(t[l]-t[l+1])
            for l in range(j+1, len(t)): w2 += (f(L2, t[l]) - T[l]) * np.abs(t[l]-t[l-1])
            print(f'I {i}: {round(w1, 3)} vs {round(w2, 3)}')
    print(DT)
    print(eDT)
    print(TP)
    print(eTP)

def molwärme():
    dt = np.array([28.34, 33.23, 43.64, 56.76, 59.18, 65.68, 75.87, 76.28, 84.91])
    edt = 0.2
    U = 5.1
    I = 1.26
    eU = 0.05
    eI = 0.01
    DQ = U * I * dt   
    eDQ = np.sqrt((I*dt*eU)**2 + (U*dt*eI)**2 + (U*I*edt)**2)
    m = 103e-3
    M = 28.085e-3
    C = M / m * DQ / DT
    eC = np.sqrt((M / m * eDQ / DT)**2  + (M / m * DQ / DT**2 * eDT)**2)
    print(C)
    print(eC)
    table = ''
    for i in range(9):
        table += '\t {:n} & \\num{{{:.2f}({:.2f})}} & \\num{{{:.2f}({:.2f})}} & \\num{{{:.2f}({:.2f})}} \\\\ \\hline\n'.format(i+1, TP[i], eTP[i], DT[i], eDT[i], C[i], eC[i])
        #table += f'\t\t {i+1} & \\num{{{round(DT[i],3)}({round(eDT[i],3)})}} & \\num{{{round(C[i],3)}({round(eC[i],3)})}} \\\\ \\hline\n'
    print(table)

#seinstein()   
molwärme()
#plots()
