import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from scipy import signal
from scipy import stats
from fractions import Fraction
from lib import helpers as hp

joe = False
eta = 1.82e-5
röl = 870
rluft = 1.293
g = 9.81
roh = g*(röl-rluft)
s = 60.0 * 16.8e-6
d = 5.0e-3
U = 510.0

def radius(v1, v2): return np.sqrt(eta*(v2-v1)/roh) * 3.0 / 2.0
def ladung(v1, v2): return 9*np.pi*d*eta*(v1+v2)*np.sqrt(eta*(v2-v1)/roh) / (2*U)    
def A(v1, v2): return eta / (roh * np.sqrt(eta*(v2-v1)/roh))
def eradius(v1, v2, ev1, ev2): return np.sqrt((A(v1,v2)*3.0/4.0)**2 * (ev1**2+ev2**2))
def eladung(v1, v2, ev1, ev2): return np.sqrt((A(v1,v2)*9*np.pi*d*eta/(4.0*U))**2 * (((3*v1-v2)*ev1)**2 + ((3*v2-v1)*ev2)**2))

def eaverage(ea): return np.sqrt(np.sum(ea**2)) / len(ea)

data = hp.excel('data/v26_messwerte.xlsx')

r = np.zeros((2,15))
er = np.zeros((2,15))
q = np.zeros((2,15))
eq = np.zeros((2,15))

def parse(j, start):
    et = 0.3
    for i in range(0, 15):
        times = data.array(f'C{start+i}:H{start+i}')
        times = np.array([times[0], *np.diff(times)])
        v = s / times
        ev = (v / times) * et
        v1 = np.array(v[1::2])
        v2 = np.array(v[::2])
        ev1 = np.array(ev[1::2])
        ev2 = np.array(ev[::2])
        r[j][i] = np.average(radius(v1, v2) * 1e6)
        er[j][i] = eaverage(eradius(v1, v2, ev1, ev2) * 1e6)
        q[j][i] = np.average(ladung(v1, v2) * 1e19)
        eq[j][i] = eaverage(eladung(v1, v2, ev1, ev2) * 1e19)
        
parse(0, 19) # ich
parse(1, 37) # jakob   
#print(q)

def plots():
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(label='Tröpfchennummer', majors=1), hp.AxisConfig(label='Radius in $\mu m$', majors=0.05, minors=2))
    ax.errorbar(range(1, 16), r[0], yerr=er[0], fmt='o', label='Joel', capsize=5)
    ax.errorbar(range(1, 16), r[1], yerr=er[1], fmt='o', label='Jakob', capsize=5)
    plt.legend()
    plt.savefig(f'tmp/v26_rad')
    #plt.clf()
    ax.cla()
    ax.config(hp.AxisConfig(label='Tröpfchennummer', majors=1), hp.AxisConfig(label='Ladung in $10^{-19}C$', majors=1, minors=2))
    ax.errorbar(range(1, 16), q[0], yerr=eq[0], fmt='o', label='Joel', capsize=5)
    ax.errorbar(range(1, 16), q[1], yerr=eq[1], fmt='o', label='Jakob', capsize=5)
    plt.legend()
    plt.savefig(f'tmp/v26_lad')

def func(n, e):
    return e * n

def index(a):
    """
    Findet ein array von ganzen zahlen [n1, n2, ...] sodass n_i * c = a_i.
    Wobei c eine konstante die durch einen linearen fit mit n_i als x werte und a_i als y werte
    """
    r = -1 # bester vergleichswert
    n = -1 # bester vielfache des ersten indizes
    for i in range(1, 10): # Eine größere Range kann zu genaueren Ergebnissen führen
        b = a[0] / i # teste vielfacher
        d = np.abs((a[1:] / b) % 1.0 - 0.5) # berechne vergleichswert; nicht optimal aber funktioniert für diesen fall
        c = np.product(d * 2) 
        if c > r: # vergleiche
            r = c # besserer wert gefunden
            n = i
    i = np.zeros(len(a))
    i[0] = n
    g = a[0] / n
    for j in range(1, len(a)): # finde alle anderen anhand des ersten vielfachen
        i[j] = int(round(a[j] / g))
    return i

def dichteplot():
    N = 300
    x = np.linspace(2, 14, N)
    D = np.zeros(N)
    Dp = np.zeros(N)
    Dm = np.zeros(N)
    w = 0.2
    qp = q + eq
    qm = q - eq
    for i in range(N):
        D[i] = np.sum(np.exp(-(x[i]-q)**2/w))
        Dp[i] = np.sum(np.exp(-(x[i]-qp)**2/w))
        Dm[i] = np.sum(np.exp(-(x[i]-qm)**2/w))
        
    peaks,_ = signal.find_peaks(D, distance=1)
    peaks_p,_ = signal.find_peaks(Dp, distance=1)
    peaks_m,_ = signal.find_peaks(Dm, distance=1)
        
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(label='Ladung q in $10^{-19}C$', majors=1, minors=2), hp.AxisConfig(label='Ladungsdichte', majors=2, minors=2, max=17, min=-0.2))
    ax.plot(x, D, label='Ladungsdichteverteilung')
    ax.plot(x, Dp, label='Ladungsdichteverteilung $+$ Fehler', alpha=0.4)
    ax.plot(x, Dm, label='Ladungsdichteverteilung $-$ Fehler', alpha=0.4)
    
    for i in peaks:
        ax.annotate(f'{round(x[i], 3)}', xy=(x[i]-0.5, D[i] + 0.1))
    for i in peaks_p:
        off = 1.1 if i == peaks_p[1] else 0.1
        ax.annotate(f'{round(x[i], 3)}', xy=(x[i]-0.5, Dp[i] + off))
    for i in peaks_m:
        ax.annotate(f'{round(x[i], 3)}', xy=(x[i]-0.5, Dm[i] + 0.1))
    
    def error_max(i):
        return round(max(abs(x[peaks[i]] - x[peaks_p[i]]), abs(x[peaks[i]] - x[peaks_m[min(1, i)]])), 3)
    
    a = [round(x[peaks[0]], 3), round(x[peaks[1]], 3), round(x[peaks[2]], 3)]
    ea = [error_max(0), error_max(1), error_max(2)]
    b = index(a)
    print(b)
    popt, pcov = optimize.curve_fit(func, b, a, sigma=ea, absolute_sigma=True)
    e = popt[0]
    ee = np.sqrt(np.diag(pcov))[0]
    print(f'{round(e, 4)} ± {round(ee, 4)}')
    
    s = '{-19}'
    ax.text_box(f'Maxima bei q in $10^{s}C$\n${a[0]}\pm{ea[0]}$\n${a[1]}\pm{ea[1]}$\n${a[2]}\pm{ea[2]}$', 0.975, 0.75, horizontal='right')
    plt.grid(which='minor', alpha=0.3)
    plt.legend()
    plt.savefig('tmp/v26_ladungsdichte.png')
    #plt.show()
    ax.cla()
    ax.config(hp.AxisConfig(label='Vielfache der Elementarladung n', majors=1), hp.AxisConfig(label='Ladung q in $10^{-19}C$', majors=1, minors=2))
    ax.plot([0, b[-1]], [0, e * b[-1]], 'k--', label=f'Lineare Regression $q(n) = e\cdot n$')
    ax.errorbar(b, a, yerr=ea, fmt='or', label='Maxima der Ladungsdichteverteilung', capsize=5)
    ax.text_box(f'e=${round(e, 4)} \pm {round(ee, 4)}$', 0.025, 0.82)
    plt.legend()
    plt.savefig('tmp/v26_millikan_elemetarladung')
    #plt.show()

#plots()
dichteplot()
