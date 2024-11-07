import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from scipy import signal
from scipy import stats
from fractions import Fraction
from lib import helpers as hp

def eaverage(ea): return np.sqrt(np.sum(ea**2)) / len(ea)

def kalibrierung(do_plot=True):
    T = np.arange(0, 150.1, 10) + 273.15
    R = [1000, 1039.0, 1077.9, 1116.7, 1155.4, 1194.0, 1232.4, 1270.7, 1308.9, 1347.0, 1385.0, 1422.9, 1460.6, 1498.2, 1535.8, 1573.0]
    
    l = stats.linregress(R, T)
    if not do_plot: return l
    m = l.slope
    b = l.intercept
    print(m, b)
    
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(label='Widerstand in $\Omega$'), hp.AxisConfig(label='Temperatur in K'))
    ax.plot([R[0], R[-1]], [m*(R[0])+b, m*(R[-1])+b], 'k--', label='Lineare Regression $T(R)=m\cdot R+b$')
    ax.plot(R, T, 'o', label='Kalibrierungsdaten')
    ax.text_box(f'$m=({round(m, 3)}\pm{round(l.stderr, 5)})K/\Omega$\n$b=({round(b, 3)}\pm{round(l.intercept_stderr, 3)})K$', 0.025, 0.82)
    plt.legend()
    #plt.show()
    plt.savefig('tmp/v27_kalib')
    return l
    
def waermeleitfaehigkeit():
    l = kalibrierung(True)
    m = l.slope
    b = l.intercept
    em = l.stderr
    eb = l.intercept_stderr
    
    rKs = np.array([19.9, 19.9, 19.8, 19.9, 19.6]) * 1e-3 / 2.0
    rAs = np.array([19.8, 19.9, 19.9, 19.9, 20.0]) * 1e-3 / 2.0
    rK = np.average(rKs)
    erK = np.std(rKs)
    rA = np.average(rAs)
    erA = np.std(rAs)
    x1 = 50 * 1e-3
    x2 = 150 * 1e-3
    x = x2-x1
    data = hp.excel('data/v27_messwerte.xlsx')
    tK = data.array('B9:B69')
    R1K = data.array('C9:C69') * 1e3
    R2K = data.array('D9:D69') * 1e3
    T1K = R1K * m + b
    T2K = R2K * m + b
    I = 0.2
    UK = 10.025
    tA = data.array('B72:B131')
    R1A = data.array('C72:C131') * 1e3
    R2A = data.array('D72:D131') * 1e3
    T1A = R1A * m + b
    T2A = R2A * m + b
    UA = 10.07
    eI = 0.005
    eU = 0.005
    eR = 1
    
    def error_T(R):
        return np.sqrt((m*eR)**2 + (R*em)**2 + eb**2)
    
    def error_l(l, U, r, er, dT, edT):
        return l*np.sqrt((eU/U)**2 + (eI/I)**2 + (2*er/r)**2 + (edT/dT)**2)
    
    fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=[12/2.54, 9/2.54])
    ax.config(hp.AxisConfig(label='Zeit t in s', majors=240, minors=4), hp.AxisConfig(label='Widerstand R in $\Omega$', minors=2))
    ax.plot(tK, R1K, '.', label='$R_1$ bei $x_1$')
    ax.plot(tK, R2K, '.', label='$R_2$ bei $x_2$')
    plt.legend()
    plt.savefig('tmp/v27_1_R_cu')
    ax.cla()
    ax.config(hp.AxisConfig(label='Zeit t in s', majors=240, minors=4), hp.AxisConfig(label='Widerstand R in $\Omega$', minors=2))
    ax.plot(tA, R1A, '.', label='$R_1$ bei $x_1$')
    ax.plot(tA, R2A, '.', label='$R_2$ bei $x_2$')
    plt.legend()
    plt.savefig('tmp/v27_1_R_al')
    ax.cla()
    ax.config(hp.AxisConfig(label='Zeit t in s', majors=240, minors=4), hp.AxisConfig(label='Temperatur T in $K$', minors=2))
    ax.errorbar(tK, T1K, yerr=error_T(R1K), fmt='.', label='$T_1$ bei $x_1$', capsize=3)
    ax.errorbar(tK, T2K, yerr=error_T(R2K), fmt='.', label='$T_2$ bei $x_2$', capsize=3)
    plt.legend()
    plt.savefig('tmp/v27_1_T_cu')
    ax.cla()
    ax.config(hp.AxisConfig(label='Zeit t in s', majors=240, minors=4), hp.AxisConfig(label='Temperatur T in $K$', minors=2))
    ax.errorbar(tA, T1A, yerr=error_T(R1A), fmt='.', label='$T_1$ bei $x_1$', capsize=3)
    ax.errorbar(tA, T2A, yerr=error_T(R2A), fmt='.', label='$T_2$ bei $x_2$', capsize=3)
    plt.legend()
    plt.savefig('tmp/v27_1_T_al')
    
    dRK = np.abs(R1K-R2K)
    #print(dRK)
    dRA = np.abs(R1A-R2A)
    dTK = dRK * m
    dTA = dRA * m
    constTK = -17
    constTA = -16
    edTK = np.full(len(dTK), m*np.sqrt(2))
    edTA = np.full(len(dTA), m*np.sqrt(2))
    cTK = np.average(dTK[constTK:])#+ 273.15
    cTA = np.average(dTA[constTA:])#+ 273.15
    ecTK = eaverage(edTK[constTK:])
    ecTA = eaverage(edTK[constTA:])
    ax.cla()
    ax.config(hp.AxisConfig(label='Zeit t in s', majors=240, minors=4), hp.AxisConfig(label='Temperatur $\delta T$ in $K$', majors=0.5, minors=2))
    ax.errorbar(tK[:constTK], dTK[:constTK], yerr=edTK[:constTK], fmt='.', label='$\delta T$ für Kupfer', capsize=3)
    ax.errorbar(tK[constTK:], dTK[constTK:], yerr=edTK[constTK:], fmt='.', label='$\delta T≈$const für Kupfer', capsize=3)
    ax.text_box(f'$\delta T≈const={round(cTK, 3)}K$', 0.975, 0.2, horizontal='right', vertical='bottom')
    plt.legend()
    plt.savefig('tmp/v27_1_dT_cu')
    ax.cla()
    ax.config(hp.AxisConfig(label='Zeit t in s', majors=240, minors=4), hp.AxisConfig(label='Temperatur $\delta T$ in $K$', majors=0.5, minors=2))
    ax.errorbar(tA[:constTA], dTA[:constTA], yerr=edTA[:constTA], fmt='.', label='$\delta T$ für Aluminium', capsize=3)
    ax.errorbar(tA[constTA:], dTA[constTA:], yerr=edTA[constTA:], fmt='.', label='$\delta T≈$const für Aluminium', capsize=3)
    ax.text_box(f'$\delta T≈const={round(cTA, 3)} K$', 0.975, 0.2, horizontal='right', vertical='bottom')
    plt.legend()
    plt.savefig('tmp/v27_1_dT_al')
    
    jK = UK * I / (np.pi * rK**2)
    jA = UA * I / (np.pi * rA**2)
    lK = jK * (x/cTK)
    lA = jA * (x/cTA)
    elK = error_l(lK, UK, rK, erK, cTK, ecTK)
    elA = error_l(lA, UA, rA, erA, cTA, ecTA)
    print(f'Wämeleitkoeff. Kupfer {lK} ± {elK}')
    print(f'Wämeleitkoeff. Aluminium {lA} ± {elA}')
    
#kalibrierung()
waermeleitfaehigkeit()
