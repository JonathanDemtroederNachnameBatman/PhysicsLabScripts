import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import scipy.signal
from matplotlib.ticker import LogLocator, FixedLocator, MultipleLocator, ScalarFormatter
from scipy import optimize
from scipy import signal
from scipy import stats

from etc import helpers as hp

def save(file, svg=True):
    plt.savefig(f'../../tmp/fprak_5_{file}.{'svg' if svg else 'png'}', bbox_inches='tight')

def inv_temp_c(R, A, B, C):
    return A * np.log(R) + B + C/np.log(R)

def widerstand_c(T, R2):
    Ta = np.delete(T, [13, 18])
    Ra = np.delete(R2, [13, 18])

    fig, ax = plt.subplots(figsize=(16/2.54, 9.5/2.54))
    ax.config(hp.AxisConfig(label='Widerstand $R_{Kohlenstoff}$ in $\\Omega$'),
              hp.AxisConfig(label='Temperatur 1/T in 1/K'))
    ax.plot(Ra, 1 / Ta, '.', label='Widerstand')
    ax.plot(R2[13], 1 / T[13], '.r', label='Ausreißer')
    ax.plot(R2[18], 1 / T[18], '.r')

    popt, pcov = optimize.curve_fit(inv_temp_c, Ra, 1 / Ta)
    print(popt)
    epopt = np.sqrt(np.diag(pcov))
    x = np.linspace(Ra[0], Ra[-1], 100)
    y = inv_temp_c(x, *popt)
    ax.plot(x, y, '--', label='Fit $1/T=A\\cdot ln(R) + B + C/ln(R)$')
    ax.grid(True)
    ax.text_box(f'$A={popt[0]:.4f}\\pm{epopt[0]:.4f}$\n$B={popt[1]:.4f}\\pm{epopt[1]:.4f}$\n$C={popt[2]:.4f}\\pm{epopt[2]:.4f}$', 0.975, 0.035, horizontal='right', vertical='bottom')
    plt.legend()
    save('kohlenstoff')
    plt.show()

def widerstand_supra(T, R):
    fig, ax = plt.subplots(figsize=(16/2.54, 9.5/2.54))
    ax.config(hp.AxisConfig(label='$R_{YBCO}$ in $\\Omega$'), hp.AxisConfig(label='T in K'))
    ax.plot(R, T, '.')
    save('htsl')
    plt.show()

    fig, ax = plt.subplots(figsize=(12 / 2.54, 9 / 2.54))
    ax.config(hp.AxisConfig(label='$R_{YBCO}$ in $\\Omega$', majors=0.04, minors=4), hp.AxisConfig(label='T in K', minors=2))
    ax.plot(R, T, '.')
    ax.set_xlim(1.4, 1.56)
    ax.set_ylim(75, 90)
    T1 = T[-2]
    T2 = T[-4]
    R1 = R[-2]
    R2 = R[-4]
    print(T1, T2, R1, R2)
    print(R[-5], R[-6])
    print(np.std([T1, T2]))
    ax.plot([R1, R[-3], R2], [T1, T[-3], T2], 'r.', label=f'Sprung, Mittelwert: ${round((T2+T1)/2, 3)}$ K')
    plt.legend()
    save('sprungtemperatur')
    plt.show()
    pass

def widerstand():
    sheet = hp.excel('../../data/fprak_5/Miristkalt.xlsx')
    D1 = sheet.array('A51:D81').T
    T0 = 18.0 + 273.15
    eT = 0.3
    TN = 77.5
    t = D1[0] * 60
    U1 = D1[1]
    U2 = D1[2]
    U3 = D1[3]
    I1 = 0.621 / 100
    I2 = 0.0925 / 100
    R1 = U1 / I1
    R2 = U2 / I2
    R3 = U3 / 0.055

    m = (TN - T0) / (R1[-1] - R1[0])
    b = T0 - m * R1[0]
    T = R1 * m + b
    #print(m)
    #print(b)
    #print(T)

    #fig, ax = plt.subplots()
    #ax.config(hp.AxisConfig(), hp.AxisConfig())
    #ax.plot(R1, T)
    #ax.grid(True)
    #plt.show()

    #widerstand_c(T, R2)
    widerstand_supra(T, R3)

def verdampfungsentalpie():
    sheet = hp.excel('../../data/fprak_5/Miristkalt.xlsx')
    D1 = sheet.array('A88:B97').T
    D2 = sheet.array('D88:E98').T
    D3 = sheet.array('G88:H98').T
    P0 = 0
    U1 = 35.0
    U2 = 49.1
    eU = 0.1
    I1 = 0.746
    I2 = 1.043
    eI = 0.001
    P1 = U1 * I1
    P2 = U2 * I2
    eP1 = np.sqrt((I1*eU)**2 + (U1*eI)**2)
    eP2 = np.sqrt((I2*eU)**2 + (U2*eI)**2)
    print(eP1, eP2)

    fig, ax = plt.subplots(figsize=(16/2.54, 9.5/2.54))
    ax.config(hp.AxisConfig(label='t in s', majors=300, minors=5), hp.AxisConfig(label='V in $m^3$', min=0.4, max=1.9, minors=2))
    l0 = stats.linregress(D1[0]*60, D1[1])
    l1 = stats.linregress(D2[0]*60, D2[1])
    l2 = stats.linregress(D3[0]*60, D3[1])
    print('r: ', l0.rvalue, l1.rvalue, l2.rvalue)
    dVdt1 = (l1.slope - l0.slope)
    dVdt2 = (l2.slope - l0.slope)
    edVdt1 = np.sqrt(l1.stderr**2 + l0.stderr**2)
    edVdt2 = np.sqrt(l2.stderr**2 + l0.stderr**2)
    print(dVdt1*1e5, edVdt1*1e5, dVdt2*1e5, edVdt2*1e5)
    x = np.array([0, D1[0][-1]]) * 60
    ax.plot(x, x * l0.slope + l0.intercept, '--', color='tab:blue')
    ax.plot(x, x * l1.slope + l1.intercept, '--', color='tab:orange')
    ax.plot(x, x * l2.slope + l2.intercept, '--', color='tab:green')
    ax.plot(D1[0]*60, D1[1], 'o', label=f'$P_0=0W$, Slope = {round(l0.slope, 6):.2e}', color='tab:blue')
    ax.plot(D2[0]*60, D2[1], 'o', label=f'$P_1={P1:.3f}W$, Slope = {round(l1.slope, 6):.2e}', color='tab:orange')
    ax.plot(D3[0]*60, D3[1], 'o', label=f'$P_2={P2:.3f}W$, Slope = {round(l2.slope, 6):.2e}', color='tab:green')

    #Vm = 15.121e-3
    Vm = 22.413e-3
    eVm = 8.770e-3
    dH1 = P1 / dVdt1
    dH2 = P2 / dVdt2
    edH1 = np.sqrt((eP1 / dVdt1)**2 + (P1 * edVdt1 / dVdt1)**2)
    edH2 = np.sqrt((eP2 / dVdt2)**2 + (P2 * edVdt2 / dVdt2)**2)
    dH = (dH1 + dH2) / 2
    edH = np.sqrt((edH1/2)**2 + (edH2/2)**2)
    print(dH1, edH1, dH2, edH2)
    print(dH, edH)
    #print(dH * Vm *1e-3, np.sqrt((edH*Vm)**2 + (dH*eVm)**2)*1e-3)
    print(dH * Vm *1e-3, edH * Vm *1e-3)
    plt.legend()
    save('verdampfen')
    plt.show()

widerstand()
#verdampfungsentalpie()
