import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from scipy import optimize
from scipy import signal
from scipy import stats
from fractions import Fraction
from lib import helpers as hp

def teil4():
    Dp = pd.read_csv('data/v28/45psteigend.txt', decimal=',', header=None, dtype=np.float64, skiprows=3, sep='\t').to_numpy().T
    Dn = pd.read_csv('data/v28/45nsteigend.txt', decimal=',', header=None, dtype=np.float64, skiprows=3, sep='\t').to_numpy().T
    #T = D[0]
    #UH = D[1]
    #UP = D[2]
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(label='$T$ in °$C$', majors=10, minors=2), hp.AxisConfig(label='$U_H$ in $mV$', majors=20, minors=2))
    ax.plot(Dp[0], -Dp[1], '.', label='p-dotiert')
    ax.plot(Dn[0], -Dn[1], '.', label='n-dotiert')
    plt.legend()
    plt.savefig('tmp/v28_4.png')
    #plt.show()
    
    Udiff = np.max(-Dp[1]) - np.min(-Dp[1])
    c = -1
    j = -1
    for i in range(len(Dp[1])):
        a = np.abs(Udiff + Dp[1][i])
        if c < 0 or a < c:
            c = a
            j = i
    print(Udiff, Dp[1][j], Dp[0][j], Dp[2][j])
    eU = 0.01
    eI = 1e-3
    def error_R(U, I):
        return np.sqrt((eU/I)**2 + (U*eI/I**2)**2)
    Ui = Dp[2][j]
    I = 30e-3
    Ri = Ui / I
    eRi = error_R(Ui, I)
    Ue = Dp[2][0]
    Re = Ue / I
    eRe = error_R(Ue, I)
    print(Ui, Ri, eRi)
    print(Ue, Re, eRe)
    mu = Re / (Re - Ri)
    emu = np.sqrt((eRe*Ri/(Re-Ri)**2)**2 + (eRi*Re/(Re-Ri)**2)**2)
    print(mu, emu)
    
def f_slope(L):
    return f'{round(L.slope, 3)}\pm{round(L.stderr, 3)}'
def f_inter(L):
    return f'{round(L.intercept, 3)}\pm{round(L.intercept_stderr, 3)}'
    
def teil3():
    Dp = pd.read_csv('data/v28/3p.txt', decimal=',', header=None, dtype=np.float64, skiprows=3, sep='\t').to_numpy().T
    Dn = pd.read_csv('data/v28/3n.txt', decimal=',', header=None, dtype=np.float64, skiprows=3, sep='\t').to_numpy().T
    
    Lp = stats.linregress(Dp[0]*1e-3, Dp[1])
    Ln = stats.linregress(Dn[0]*1e-3, Dn[1])
    Rp = Lp.slope
    Rn = Ln.slope
    l = 20e-3
    A = 10e-3 * 1e-3
    sp = l / (Rp*A)
    sn = l / (Rn*A)
    esp = Lp.stderr * l / (A*Rp**2)
    esn	                                                     = Ln.stderr * l / (A*Rn**2)
    print(sp, esp)
    print(sn, esn)
    
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(label='Probenstrom $I$ in A', majors=0.015, minors=3), hp.AxisConfig(label='Probenspannung $U$ in $V$', majors=1.5, minors=3))
    ax.plot(Dp[0]*1e-3, Dp[1], 'o', label='p-dotiert')
    ax.plot(Dn[0]*1e-3, Dn[1], 'o', label='n-dotiert')
    ax.plot(Dp[0]*1e-3, Dp[0]*1e-3 * Lp.slope + Lp.intercept, 'r--', label='p-dotiert linearer fit $U_p = R_p\cdot I_p+b_p$')
    ax.plot(Dn[0]*1e-3, Dn[0]*1e-3 * Ln.slope + Ln.intercept, 'k--', label='n-dotiert linearer fit $U_n = R_n\cdot I_n+b_n$')
    ax.text_box(f'$R_p={f_slope(Lp)}$\n$b_p={f_inter(Lp)}$\n$R_n={f_slope(Ln)}$\n$b_n={f_inter(Ln)}$', 0.97, 0.03, vertical='bottom', horizontal='right')
    plt.legend()
    plt.savefig('tmp/v28_3.png')
    plt.show()
    
teil3()
#teil4()
