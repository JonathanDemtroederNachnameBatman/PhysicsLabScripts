import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from scipy import optimize
from scipy import signal
from scipy import stats
from fractions import Fraction
from lib import helpers as hp

gauß = 0
lorenz = 1
pseudo_voigt = 2

gauß_eq = r'$A\cdot e^{\frac{-4 ln(2) (x-a)^2}{\sigma^2}}+B$'
lorenz_eq = r'$A\cdot \frac{\sigma^2}{4 (x-a)^2+\sigma^2}$'
pseudo_voigt_eq = f'$\\eta {gauß_eq[1:-1]} + (1-\\eta){lorenz_eq[1:-1]}$' 

def gauß_func(x, A, B, sigma, a):
    return A * np.exp((-4*np.log(2)*(x-a)**2)/sigma**2) + B

def lorenz_func(x, A, B, sigma, a):
    return A * sigma**2 / (4*(x-a)**2 + sigma**2) + B

def pseudo_voigt_func(x, A, B, sigma, a, n):
    return n*lorenz_func(x, A, B, sigma, a) + (1-n)*gauß_func(x, A, B, a, n)

def get_func(i):
    if i == 0: return gauß_func
    elif i == 1: return lorenz_func
    else: return pseudo_voigt_func
    
def get_eq(i):
    if i == 0: return 'Gauß'
    elif i == 1: return 'Lorenz'
    else: return 'Pseudo-Voigt'

def doSalz():
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(label=r'2$\vartheta$ in °'), hp.AxisConfig(label='Impulse'))
    D = pd.read_csv(f'data/fprak_1/241018_Salz.txt', decimal='.', header=None, dtype=np.float64, skiprows=5, sep='   ').to_numpy().T

    ax.plot(D[0], D[1], '.', markersize=1)

    plt.show()

def doPeak(fein: bool, i: int, func_type: int, guess):
    sfein = "Fein" if fein else "Grob"
    D = pd.read_csv(f'data/fprak_1/241018_Si_{sfein.lower()}_Peak{i}.txt', decimal='.', header=None, dtype=np.float64, skiprows=5, sep='\s+').to_numpy().T
    x = D[0]
    y = D[1]
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(label=r'2$\vartheta$ in °', minors=2), hp.AxisConfig(label='counts', minors=2))
    
    if guess:
        func = get_func(func_type)
        popt, pcov = optimize.curve_fit(func, x, y, guess, bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, 1]), max_nfev=2000)
        print(popt)
        X = np.linspace(x[0], x[-1], 400)
        Y = func(X, *popt)
        ax.plot(X, Y, label='Fit für ' + get_eq(func_type) + ' Peak')
    
    ax.plot(x, y, '.k', markersize=2, label=f'Silizium, {sfein}, Peak {i}')
    if guess:
        ax.text_box(hp.param_text_fit(['A', 'B', r'\sigma', 'a', '\\eta'], popt, pcov, 3), 0.975, 0.80, horizontal='right')

    plt.legend()
    plt.savefig(f'tmp/fprak_1_si_{sfein.lower()}_{i}.png')
    
def doPeaks():
    doPeak(True, 1, pseudo_voigt, [1000, 200, 1, 30, 0.5])
    doPeak(True, 2, pseudo_voigt, [400, 100, 1, 46, 0.5])
    doPeak(True, 3, pseudo_voigt, [200, 50, 1, 55, 0.5])
    doPeak(False, 1, pseudo_voigt, [9000, 1, 1, 28, 0.5])
    doPeak(False, 2, pseudo_voigt, [5000, 1, 1, 46.80, 0.5])
    doPeak(False, 3, pseudo_voigt, [2500, 1, 1, 55, 0.5])

def doSalz():
    D = pd.read_csv(f'data/fprak_1/241018_Salz.txt', decimal='.', header=None, dtype=np.float64, skiprows=5, sep='\s+').to_numpy().T
    x = D[0]
    y = D[1]
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(16*cm, 9*cm))
    ax.config(hp.AxisConfig(label=r'2$\vartheta$ in °', minors=2), hp.AxisConfig(label='counts', minors=2))
    ax.plot(x, y, 'k', markersize=2, label='Salz')
    
    plt.legend()
    plt.savefig('tmp/fprak_1_salz.png')

doSalz()
