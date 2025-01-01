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

def read(file, skiprows=1, decimal='.'):
    return pd.read_csv(f'../../data/fprak_4/{file}.txt', decimal=decimal, header=None, dtype=np.float64, skiprows=skiprows, sep=r'\s+').to_numpy().T

def save(file, svg=True):
    plt.savefig(f'../../tmp/fprak_4_{file}.{'svg' if svg else 'png'}', bbox_inches='tight')

def save_arr(a, file):
    f = f'../../data/fprak_4/{file}.txt'
    np.savetxt(f, a.T, fmt=['%f', '%f'], delimiter='\t')

def druck():
    D = read('druck')
    t = D[0]
    p = D[1]
    fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xticks([30, 45, 60, 90, 120, 180, 240, 300, 360, 480])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.config(hp.AxisConfig(label='t in s'), hp.AxisConfig(label='Druck in mbar'))
    ax.plot(t, p, '.')
    save('druck')
    plt.show()

def langmuir(name):
    D = read(name)
    p = np.average(D[1])
    U = D[0]
    I = D[2]
    fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))
    ax.config(hp.AxisConfig(label='Sondenspannung $U_B$ in $V$'), hp.AxisConfig(label='Strom $I$ in $mA$'))
    ax.plot(U, I, '.k', label=name, markersize=1)
    ax.text_box(f'$p={round(p, 4)} mb$', 0.025, 0.85)
    ax.legend()
    save(name, svg=True)
    plt.show()
    fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))
    ax.config(hp.AxisConfig(label='Sondenspannung $U_B$ in $V$'), hp.AxisConfig(label='Strom $I$ in $mA$'))
    ax.plot(U, I, '.k', label=f'{name} - ln', markersize=1)
    ax.text_box(f'$p={round(p, 4)} mb$', 0.025, 0.85)
    ax.set_yscale('log')
    ax.legend()
    save(f'{name}_ln', svg=True)

def plot_linregress(ax, x, y, l, fmt='--', linewidth=2.0):
    x1 = min(max((np.min(y)-l.intercept)/l.slope, np.min(x)), np.max(x))
    x2 = min(max((np.max(y)-l.intercept)/l.slope, np.min(x)), np.max(x))
    x0 = np.array([x1, x2])
    ax.plot(x0, x0 * l.slope + l.intercept, fmt, linewidth=linewidth)

def analyze_langmuir(name, k, i1, i2, j1, j2):
    D = read(name)
    p = np.average(D[1])
    U = D[0]
    I = D[2]
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    ax.config(hp.AxisConfig(label='Sondenspannung $U_B$ in $V$'), hp.AxisConfig(label='Strom $log_{10}(I)$ in $log_{10}(mA)$'))
    #ax.text_box(f'$p={round(p, 3)}\\pm{round(np.std(D[1]), 3)} mb$', 0.025, 0.82)

    to_delete = np.where(I <= 0)
    lU = np.delete(U, to_delete)
    lI = np.log(np.delete(I, to_delete))
    print(len(lU))
    l1 = stats.linregress(lU[i1:i2], lI[i1:i2])
    l2 = stats.linregress(lU[j1:j2], lI[j1:j2])
    plot_linregress(ax, lU, lI, l1, linewidth=1.5)
    plot_linregress(ax, lU, lI, l2, linewidth=1.5)
    # schnittpunkt (Plasmapotential, Elektronensättigungsstrom)
    Vp = (l1.intercept - l2.intercept) / (l2.slope-l1.slope)
    Ies = np.exp(Vp * l1.slope + l1.intercept)
    #print(Vp, Ies, 10**(Vp * l2.slope + l2.intercept))

    ax.plot(lU[:i1], lI[:i1], '.k', label=name, markersize=1)
    ax.plot(lU[i2:j1], lI[i2:j1], '.k', markersize=1)
    ax.plot(lU[i1:i2], lI[i1:i2], '.', markersize=1)
    ax.plot(lU[j1:j2], lI[j1:j2], '.', markersize=1)
    save('ui_auswertung', svg=True)
    plt.show()

    # normal for Iis
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    ax.config(hp.AxisConfig(label='Sondenspannung $U_B$ in $V$'),
              hp.AxisConfig(label='Strom $I$ in $mA$'))
    l = stats.linregress(U[:k], I[:k])
    Iis = Vp * l.slope + l.intercept
    plot_linregress(ax, U, I, l, linewidth=1.5)
    ax.plot(U[:k], I[:k], '.', markersize=1)
    ax.plot(U[k:], I[k:], '.k', markersize=1)
    x = np.full(2, Vp)
    ax.plot(x, [np.min(I), np.max(I)], '--', label='$U_p$')
    plt.legend()
    save('ui_auswertung_ln', svg=True)
    plt.show()
    i = np.argmin(np.abs(I))
    Vf = U[i]
    print(f'Vp  = {Vp}')
    print(f'Vf  = {Vf}')
    print(f'Ies = {Ies}')
    print(f'Iis = {Iis}')
    Te = 1/l1.slope
    #Te2 = (Vf-Vp)/np.log(Iis/Ies)
    #Te3 = (10 - 1) / np.log(np.exp(10*l1.slope+l1.intercept)/np.exp(1*l1.slope+l1.intercept))
    print(f'Te = {Te}')
    me = 9.109e-31
    mi = 14.0067 * 1.6605e-27
    Vf2 = Vp + Te*np.log(0.6*np.sqrt(2*np.pi*(me/mi))) # incorrect result
    print(f'Vf2 = {Vf2}')

def langmuir_all():
    langmuir('1')
    langmuir('mittig')
    langmuir('ganzlinks')
    langmuir('dollerechts')
    langmuir('bisschenrechts')
    langmuir('bisschenlinks')
    langmuir('2rausmitte')
    langmuir('2rausrechts')
    langmuir('4rausmitte')
    langmuir('4rausrechts')
    langmuir('7W2rausmitte')
    langmuir('7W2rausmitte_wenig_druck')
    langmuir('7W2rausrechts')
    langmuir('7W2rausrechts_wenig_druck')
    langmuir('7W4rausmitte_wenig_druck')

#druck()
#langmuir_all()
#langmuir('7W2rausmitte_wenig_druck')
analyze_langmuir('7W4rausmitte_wenig_druck', 70, 70, 160, -80, -1)
