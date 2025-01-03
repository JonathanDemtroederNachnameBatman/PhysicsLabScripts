import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import scipy.signal
from matplotlib.ticker import LogLocator, FixedLocator, MultipleLocator, ScalarFormatter
from scipy import optimize
from scipy import signal
from scipy import stats
from si_prefix import si_format

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
    #ax.text_box(f'$p={round(p, 4)} mb$', 0.025, 0.85)
    ax.legend()
    save(name, svg=False)
    plt.show()
    fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))
    ax.config(hp.AxisConfig(label='Sondenspannung $U_B$ in $V$'), hp.AxisConfig(label='Strom $I$ in $mA$'))
    ax.plot(U, I, '.k', label=f'{name} - ln', markersize=1)
    ax.text_box(f'$p={round(p, 4)} mb$', 0.025, 0.85)
    ax.set_yscale('log')
    ax.legend()
    save(f'{name}_ln', svg=False)

def plot_linregress(ax, x, y, l, fmt='--', linewidth=2.0, name=None):
    x1 = min(max((np.min(y)-l.intercept)/l.slope, np.min(x)), np.max(x))
    x2 = min(max((np.max(y)-l.intercept)/l.slope, np.min(x)), np.max(x))
    x0 = np.array([x1, x2])
    ax.plot(x0, x0 * l.slope + l.intercept, fmt, linewidth=linewidth, label=name)

def analyze_langmuir(name, k, i1, i2, j1, j2, val_only=False):
    D = read(name)
    p = np.average(D[1])
    U = D[0]
    I = D[2]
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    ax.config(hp.AxisConfig(label='Sondenspannung $U_B$ in $V$'), hp.AxisConfig(label='Strom $ln(I)$ in $ln(mA)$'))
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
    eVp = np.sqrt((l1.intercept_stderr / (l2.slope - l1.slope)) ** 2 + (l2.intercept_stderr / (l2.slope - l1.slope)) ** 2 + ((l1.intercept - l2.intercept) * l2.stderr / (l2.slope-l1.slope)**2)**2 + ((l1.intercept - l2.intercept) * l1.stderr / (l2.slope-l1.slope)**2)**2)
    Ies = np.exp(Vp * l1.slope + l1.intercept)
    eIes = np.sqrt((l1.slope * eVp)**2 + (Vp * l1.stderr)**2 + l1.intercept_stderr**2)
    #print(Vp, Ies, 10**(Vp * l2.slope + l2.intercept))
    Te = 1 / l1.slope
    eTe = l1.stderr / l1.slope

    ax.plot(lU[:i1], lI[:i1], '.k', label=name, markersize=1)
    ax.plot(lU[i2:j1], lI[i2:j1], '.k', markersize=1)
    ax.plot(lU[i1:i2], lI[i1:i2], '.', markersize=1)
    ax.plot(lU[j1:j2], lI[j1:j2], '.', markersize=1)
    #ax.plot([13.2468, 13.2468], [0, -8], '--')
    ax.text_box(f'$U_p = ({round(Vp, 3)}\\pm {round(eVp, 3)})V$\n$I_{{es}} = ({round(Ies, 3)}\\pm {round(eIes, 3)})mA$\n$T_{{e,eV}} = ({round(Te, 3)}\\pm {round(eTe, 3)})eV$', 0.97, 0.1, horizontal='right', vertical='bottom')
    save('ui_auswertung_ln', svg=True)
    plt.show()

    # normal for Iis
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    ax.config(hp.AxisConfig(label='Sondenspannung $U_B$ in $V$'),
              hp.AxisConfig(label='Strom $I$ in $mA$'))
    l = stats.linregress(U[:k], I[:k])
    Iis = Vp * l.slope + l.intercept
    eIis = np.sqrt((l.slope * eVp) ** 2 + (Vp * l.stderr) ** 2 + l.intercept_stderr ** 2)
    plot_linregress(ax, U, I, l, linewidth=1.5, name='$I_i$')
    ax.plot(U[:k], I[:k], '.', markersize=1)
    ax.plot(U[k:], I[k:], '.k', markersize=1, label='Messdaten')
    x = np.full(2, Vp)
    ax.plot(x, [np.min(I), np.max(I)], '--', label='$U_p$')
    ax.text_box(f'$I_{{is}} = ({round(Iis, 3)}\\pm {round(eIis, 3)})mA$', 0.025, 0.70)
    plt.legend()
    save('ui_auswertung', svg=True)
    plt.show()
    i = np.argmin(np.abs(I))
    Vf = U[i]
    print(f'Vp  = {Vp}')
    print(f'Vf  = {Vf}')
    print(f'Ies = {Ies}')
    print(f'Iis = {Iis}')
    #Te2 = (Vf-Vp)/np.log(Iis/Ies)
    #Te3 = (10 - 1) / np.log(np.exp(10*l1.slope+l1.intercept)/np.exp(1*l1.slope+l1.intercept))
    print(f'Te = {Te}')
    me = 9.109e-31
    mi = 14.0067 * 1.6605e-27
    e = 1.602e-19
    kB = 1.381e-23
    As = 20e-4
    e0 = 8.854e-12
    #Vf2 = Vp + Te*np.log(0.6*np.sqrt(2*np.pi*(me/mi))) # incorrect result
    #print(f'Vf2 = {Vf2}')
    ni = (5*Iis*1e-3)/(3*e*As)*np.sqrt(mi/(Te*e))
    eni = np.sqrt(((5*eIis*1e-3)/(3*e*As)*np.sqrt(mi/(Te*e)))**2 + ((5*Iis*1e-3*eTe)/(6*e*As)*np.sqrt(mi/(Te**3 * e)))**2)
    ne = (4*Ies*1e-3)/(As*e)*np.sqrt(np.pi*me/(8*Te*e))
    ene = np.sqrt(((4*eIes*1e-3)/(As*e)*np.sqrt(np.pi*me/(8*Te*e)))**2 + ((2*Ies*1e-3*eTe)/(As*e)*np.sqrt(np.pi*me/(8*Te**3 * e)))**2)
    lDe = np.sqrt(e0*Te/(ne*e))
    elDe = np.sqrt((eTe*np.sqrt(e0/(ne*e*Te))/2)**2 + (ene*np.sqrt(e0*Te/(ne**3 * e))/2)**2)
    wPe = np.sqrt(ne*e*e/(e0*me))
    ewPe = ene * np.sqrt(e*e/(ne*e0*me))/2
    g = 1 / (ne*lDe**3)
    eg = np.sqrt((ene / (ne*ne*lDe**3))**2 + (3*elDe / (ne*lDe**4))**2)
    #print(f'ni  = {si_format(ni, precision=5)}')
    #print(f'ne  = {si_format(ne, precision=5)}')
    #print(f'lDe = {si_format(lDe, precision=5)}')
    #print(f'wPe = {si_format(wPe, precision=5)}')
    #print(f'g   = {si_format(g, precision=5)}')
    col0 = ['$n_i$', '$n_e$', '$T_e$', '$\\lambda_{D,e}$', '$\\omega_{P,e}$', '$g$']
    a = [ni, ne, Te, lDe, wPe, g]
    #print(hp.to_latex_table(a, col0=col0))
    table = """
        \\hline
        $n_i$ in $\\si{{\\per\\cm\\tothe{{3}}}}$ & {} \\\\ \\hline
        $n_e$ in $\\si{{\\per\\cm\\tothe{{3}}}}$ & {} \\\\ \\hline
        $T_e$ in \\si{{\\electronvolt}} & {} \\\\ \\hline
        $\\lambda_{{D,e}}$ in \\si{{\\mm}} & {} \\\\ \\hline
        $\\omega_{{P,e}}$ in \\si{{\\giga\\hertz}} & {} \\\\ \\hline
        $g$ & {} \\\\ \\hline
    """
    #print(table.format([ni * 1e-15, ne * 1e-15, Te, lDe * 1e-3, wPe * 1e-9, g]))
    print(table.format(hp.latex_num(ni * 1e-6, power=6, err=eni * 1e-6), hp.latex_num(ne * 1e-6, power=6, err=ene * 1e-6), hp.latex_num(Te, err=eTe), hp.latex_num(lDe * 1e3, err=elDe * 1e3), hp.latex_num(wPe * 1e-9, err=ewPe * 1e-9), hp.latex_num(g, power=-7, err=eg)))
    print((0.00716 - l.intercept) / l.slope)

    fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))
    ax.config(hp.AxisConfig(label='Sondenspannung $U_B$ in V'), hp.AxisConfig(label='$dI/dU$'))
    dU = 0.15
    dIdU = np.gradient(I, dU)
    m = U[np.argmax(dIdU)]
    ax.plot(U, dIdU, '.k', markersize=1)
    ax.plot([m, m], [np.min(dIdU), np.max(dIdU)], '--', label=f'$U_p = {m}V$')
    plt.legend()
    save('didu', svg=True)
    plt.show()
    print(m)



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
#analyze_langmuir('7W2rausmitte_wenig_druck', 70, 70, 150, -70, -1)
