import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from scipy import optimize
from scipy import signal
from scipy import stats
from fractions import Fraction

from etc import helpers as hp

gauß = 0
lorenz = 1
pseudo_voigt = 2
all_models = -1

gauß_eq = r'$A\cdot e^{\frac{-4 ln(2) (x-a)^2}{\sigma^2}}+B$'
lorenz_eq = r'$A\cdot \frac{\sigma^2}{4 (x-a)^2+\sigma^2}$'
pseudo_voigt_eq = f'$\\eta {gauß_eq[1:-1]} + (1-\\eta){lorenz_eq[1:-1]}$'

orig_peaks = np.average(np.array([[28.44732, 28.51767], [47.31131, 47.43290], [56.13278, 56.28081]]), axis=1)
shift = []
peak_shift = 0.7031729707143283

d1_hkl = np.zeros(3)
d2_hkl = np.zeros(3)

def gauss_model(x, sigma, a):
    return np.exp((-4 * np.log(2) * (x - a) ** 2) / sigma ** 2)

def gauss_func(x, m, b, sigma, a):
    return m * gauss_model(x, sigma, a) + b

def lorenz_model(x, sigma, a):
    return sigma**2 / (4*(x-a)**2 + sigma**2)

def lorenz_func(x, m, b, sigma, a):
    return m * lorenz_model(x, sigma, a) + b

def pseudo_voigt_func(x, m, b, sigma, a, n):
    return m * (n*lorenz_model(x, sigma, a) + (1-n)*gauss_model(x, a, n)) + b

def get_func(i):
    if i == 0: return gauss_func
    elif i == 1: return lorenz_func
    else: return pseudo_voigt_func
    
def get_eq(i):
    if i == 0: return 'Gauß'
    elif i == 1: return 'Lorenz'
    else: return 'Pseudo-Voigt'

def plot_fit(ax, x, y, func_type: int, dashed=False):
    func = get_func(func_type)
    max1 = np.max(y)
    min_bound = [0, 0, 0, x[0], 0]
    max_bound = [max1 * 2, max1, 100, x[-1], 1]
    guess = [max1, 1, 1, np.average([x[0], x[-1]]), 0.5]
    if func_type == 0 or func_type == 1:
        guess = guess[0:-1]
        min_bound = min_bound[0:-1]
        max_bound = max_bound[0:-1]
    popt, pcov = optimize.curve_fit(func, x, y, guess, bounds=(min_bound, max_bound), max_nfev=2000)
    r = hp.calc_r(func, x, y, popt)
    r = np.sqrt(r)
    X = np.linspace(x[0], x[-1], 400)
    Y = func(X, *popt)
    #print(popt[0] + popt[1], np.max(Y), max1)
    style = '--' if dashed else '-'
    ax.plot(X, Y, style, label=f'Fit für {get_eq(func_type)} Peak')
    i_max = np.argmax(Y)
    half_max = (Y[i_max]) / 2.0 # half of maximum
    x1 = np.argmin(np.abs(Y[0:i_max] - half_max)) # closest to half max on left
    x2 = np.argmin(np.abs(Y[i_max+1:-1] - half_max)) # closest to half max on right
    return popt, pcov, r, X[x2+i_max+1] - X[x1]

def doPeak(fein: bool, i: int, func_type: int, fit=True):
    sfein = "Fein" if fein else "Grob"
    D = pd.read_csv(f'../../data/fprak_1/241018_Si_{sfein.lower()}_Peak{i}.txt', decimal='.', header=None, dtype=np.float64, skiprows=5, sep=r'\s+').to_numpy().T
    x = D[0] + peak_shift
    y = D[1]
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(16*cm, 9*cm))
    ax.config(hp.AxisConfig(label=r'2$\theta$ in °', minors=2), hp.AxisConfig(label='counts', minors=2))

    rs = np.zeros(3)
    if fit:
        if func_type < 0:
            for j in range(0, 3):
                popt, pcov, r, fwhm = plot_fit(ax, x, y, j, dashed=j==2)
                rs[j] = r
        else:
            popt, pcov, r, fwhm = plot_fit(ax, x, y, func_type)
            #print(np.deg2rad(fwhm))
            K = 0.89
            l = 1.542e-1
            el = 0.002e-1
            rfwhm = np.deg2rad(fwhm)
            erfwhm = 0.05 * rfwhm
            a = np.deg2rad(popt[3] / 2)
            ea = np.deg2rad(np.sqrt(np.diag(pcov))[3] / 2)
            d = K * l / (rfwhm * np.cos(a))
            ed = np.sqrt((K*el/(rfwhm*np.cos(a)))**2 + (erfwhm*K*l/(rfwhm**2 *np.cos(a)))**2 + (np.sin(a)*K*el/(rfwhm*np.cos(a)**2))**2)
            print(f'Korngröße ({sfein},{i}): {hp.si(d, ed, 3, ['nm'])}')
            if fein:
                d1_hkl[i-1] = d
            else:
                d2_hkl[i-1] = d
            #shift.append(orig_peaks[i-1] - popt[3])
    
    ax.plot(x, y, '.k--', markersize=2, linewidth=0.6, label=f'Silizium, {sfein}, Peak {i}')
    if fit and func_type >= 0:
        text = (f'$R={round(r,3)}$\n'
                + hp.param_text_fit(['A', 'B', r'\sigma', 'a', '\\eta'], popt, pcov, 3)
                + f'\n$FWHM={round(fwhm, 3)}$')
        ax.text_box(text, 0.975, 0.78, horizontal='right')
    else:
        ax.text_box(f'Gauß: $R={round(rs[0], 3)}$\nLorentz: $R={round(rs[1], 3)}$\nPV: $R={round(rs[2], 3)}$', 0.975, 0.63, horizontal='right')

    model = '_all' if func_type < 0 else '_pv'
    plt.legend()
    plt.savefig(f'../../tmp/fprak_1_si_{sfein.lower()}_{i}{model}.png')
    
def doPeaks():
    model = pseudo_voigt
    doPeak(True, 1, model)
    doPeak(True, 2, model)
    doPeak(True, 3, model)
    doPeak(False, 1, model)
    doPeak(False, 2, model)
    doPeak(False, 3, model)
    print(f'Korngröße fein: {round(np.average(d1_hkl), 3)}({round(np.std(d1_hkl), 3)})')
    print(f'Korngröße grob: {round(np.average(d2_hkl), 3)}({round(np.std(d2_hkl), 3)})')

def doSalz():
    D = pd.read_csv(f'../../data/fprak_1/241018_Salz.txt', decimal='.', header=None, dtype=np.float64, skiprows=5, sep=r'\s+').to_numpy().T
    x = D[0] + peak_shift
    y = D[1]
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(16*cm, 9*cm))
    ax.config(hp.AxisConfig(label=r'2$\theta$ in °', minors=2), hp.AxisConfig(label='counts', minors=2))
    ax.plot(x, y, 'k', markersize=2, label='Salz')
    
    plt.legend()
    plt.savefig('../../tmp/fprak_1_salz.png')
    #plt.show()

    maxs = optimize.m

doPeaks()
#doSalz()
