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
    x = np.linspace(Ra[0], Ra[-1], 100)
    y = inv_temp_c(x, *popt)
    ax.plot(x, y, '--', label='Fit $A\\cdot ln(R) + B + C/ln(R)$')
    ax.grid(True)
    plt.legend()
    save('kohlenstoff')
    plt.show()

def widerstand_supra(T, R):
    fig, ax = plt.subplots(figsize=(16/2.54, 9.5/2.54))
    ax.config(hp.AxisConfig(label='T'), hp.AxisConfig(label='R'))
    ax.plot(T, R, '.')

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
    print(m)
    print(b)
    #print(T)

    #fig, ax = plt.subplots()
    #ax.config(hp.AxisConfig(), hp.AxisConfig())
    #ax.plot(R1, T)
    #ax.grid(True)
    #plt.show()

    #widerstand_c(T, R2)
    widerstand_supra(T, R3)


widerstand()