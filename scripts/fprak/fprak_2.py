from cProfile import label

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import scipy.signal
from scipy import optimize
from scipy import signal
from scipy import stats
from etc import helpers as hp

def read(file):
    return pd.read_csv(f'../../data/fprak_2/{file}.txt', decimal=',', header=None, dtype=np.float64, skiprows=1, sep=r'\s+').to_numpy().T

def save(file):
    plt.savefig(f'../../tmp/fprak_2_{file}.svg')

def do_piezo():
    D = read('Laser')

    D[1] -= np.mean(D[1])
    maxs = signal.find_peaks(D[1])[0]
    #print(maxs)
    def plot_interf():
        fig, ax = plt.subplots(figsize=(16/2.54, 8/2.54))
        ax.config(hp.AxisConfig(label='Spannung in V'), hp.AxisConfig(label='Intensität'))
        ax.plot(D[0], D[1], '.k--', linewidth=1)
        ax.plot(D[0][maxs], D[1][maxs], '.r', linewidth=1)
        save('laser')
        #plt.show()

    plot_interf()

    fig, ax = plt.subplots(figsize=(16/2.54, 8/2.54))
    ax.config(hp.AxisConfig(label='Peaknummer n'), hp.AxisConfig(label='Spannung in V'))
    ns = range(1, len(maxs)+1)
    fit = stats.linregress(ns, D[0][maxs])
    fit_x = np.array([-1, ns[-1]+1])
    ax.plot(fit_x, fit_x * fit.slope + fit.intercept, 'k--', label='Linearer Fit $U=n\\cdot m + b$')
    ax.plot(ns, D[0][maxs], '.', label='Position der Peaks')
    ax.text_box(f'm={round(fit.slope, 5)} V\nb={round(fit.intercept, 5)} V\n$R^2={round(fit.rvalue, 5)}$', 0.025, 0.75)
    plt.legend()
    save('laser_kalib')
    #plt.show()

def do_fft(D, window_func=None, from_index=0, to_index=60):
    if window_func is None:
        #window_func = np.blackman(len(D[0]))
        window_func = np.full(len(D[0]), 1)
    diff = np.average(np.diff(D[0]))
    N = len(D[0])
    y = (D[1] - np.mean(D[1])) * window_func
    y = np.fft.fft(y)
    y = 2.0 / N * np.abs(y[:N // 2])
    freq = np.fft.fftfreq(N, d=diff)[:N // 2]
    return freq[from_index:to_index], y[from_index:to_index]

def voltfreq_to_lamda(voltfreq):
    # laser wellenlänge in nm mal peak position nach fft
    return 632.8 * 4.344919786096256 / voltfreq

def do_laser():
    D = read('Laser')
    N = len(D[0])


    def do_simple():
        freq, y1 = do_fft(D, np.full(N, 1), from_index=0, to_index=510)
        fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
        ax.config(hp.AxisConfig(label='$V^{-1}$'), hp.AxisConfig(label='Intensität', ticks=False))
        lw = 1
        ax.plot(freq, y1, label='Spektrum ohne Apodisierung', linewidth=lw)
        save('laser_spectrum')
        plt.show()

    do_simple()

    freq, y1 = do_fft(D, np.full(N, 1))
    freq, y2 = do_fft(D, np.hanning(N))
    freq, y3 = do_fft(D, np.hamming(N))
    freq, y4 = do_fft(D, np.bartlett(N))
    freq, y5 = do_fft(D, np.blackman(N))
    freq, y6 = do_fft(D, np.kaiser(N, 5))

    maxs = signal.find_peaks(y5, height=0.4)[0]
    #print(maxs, freq[39])

    fig, ax = plt.subplots(figsize=(16/2.54, 8/2.54))
    ax.config(hp.AxisConfig(label='$V^{-1}$'), hp.AxisConfig(label='Intensität', ticks=False, min=-0.05, max=1.15))
    lw = 1
    l = freq
    ax.plot(l, y1 + 0.3, label='Spektrum ohne Apodisierung', linewidth=lw)
    #ax.plot(freq[4:lim], y2[4:lim], label='Hanning', linewidth=lw)
    ax.plot(l, y3 + 0.2, label='Hamming', linewidth=lw)
    ax.plot(l, y4 + 0.1, label='Bartlett', linewidth=lw)
    ax.plot(l, y5 + 0.0, label='Blackman', linewidth=lw)
    ax.plot([l[maxs[0]], l[maxs[0]]], [-0.5, 1.5], 'k--', linewidth=1, label='Maxima bei $4.34492 V^{-1}$')
    #ax.plot(freq[4:lim], y6[4:lim], label='Kaiser', linewidth=lw)
    plt.legend()
    save('laser_apod')
    plt.show()

def do_halogen():
    D = read('Halogen2')

    freq, y = do_fft(D, from_index=7)
    fig, ax = plt.subplots(figsize=(16/2.54, 8/2.54))
    ax.config(hp.AxisConfig(label='$\\lambda$ in nm'), hp.AxisConfig(label='Intensität', ticks=False))
    l = voltfreq_to_lamda(freq)

    ax.plot(l, y, '.--', label='Halogen', linewidth=1)
    i = np.argmax(y)
    #ax.plot_max(l[i], y[i], f'Halogen Max: {int(l[i])}nm')
    D = read('Halogen_PbS')
    freq, y = do_fft(D, from_index=7)
    ax.plot(l, y, '.--', label='Halogen - PbS', linewidth=1)
    i = np.argmax(y)
    #ax.plot_max(l[i], y[i], f'Halogen - PbS Max: {int(l[i])}nm')

    plt.legend()
    save('halogen')
    plt.show()

    D = read('Halogen2')
    D1 = read('Halogen_blau')
    D2 = read('Halogen_gelb')
    D3 = read('Halogen_rot')
    D4 = read('Halogen_interferenz')

    fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
    ax.config(hp.AxisConfig(label='$\\lambda$ in nm'), hp.AxisConfig(label='Intensität', ticks=False))
    freq, y = do_fft(D, from_index=17)
    freq, y1 = do_fft(D1, from_index=17)
    freq, y2 = do_fft(D2, from_index=17)
    freq, y3 = do_fft(D3, from_index=17)
    freq, y4 = do_fft(D4, from_index=17)
    l = voltfreq_to_lamda(freq)
    ax.plot(l, y4, '^k--', label='Halogen - Interf', linewidth=1, alpha=0.5)
    ax.plot(l, y1, '.b--', label='Halogen - Blau', linewidth=1)
    ax.plot(l, y2, '.y--', label='Halogen - Gelb', linewidth=1)
    ax.plot(l, y3, '.r--', label='Halogen - Rot', linewidth=1)
    ax.plot(l, y, '.k--', label='Halogen', linewidth=1)
    plt.legend()
    save('halogen_color')
    plt.show()

#do_piezo()
#do_laser()
do_halogen()
