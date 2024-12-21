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

def save(file):
    plt.savefig(f'../../tmp/fprak_4_{file}.svg', bbox_inches='tight')

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

druck()
