import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
from scipy import stats
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, Locator)
import numbers
import xlwings as xw

# helper class to easily decorate multiple plots with similar data
class AxisConfig:

    def __init__(self, min=None, max=None, minors=None, majors=None, label=None, ticks=True):
        self.min = min
        self.max = max
        self.minors = minors
        self.majors = majors
        self.label = label
        self.ticks = ticks

    def apply(self, axis, set_label):
        if self.majors: 
            if isinstance(self.majors, numbers.Number):
                axis.set_major_locator(MultipleLocator(self.majors))
            elif isinstance(self.majors, Locator):
                axis.set_major_locator(self.majors)
            else: 
                raise Exception('Wrong majors type')
        if self.minors: 
            if isinstance(self.minors, numbers.Number):
                axis.set_minor_locator(MultipleLocator(self.minors))
            elif isinstance(self.minors, Locator):
                axis.set_minor_locator(self.minors)
            else: 
                raise Exception('Wrong minors type')
        if self.label: set_label(self.label, fontsize=10, fontweight='medium')
        if not self.ticks: axis.set_ticklabels([])

# helper method to easily decorate a plot
def config_axis(ax, x: AxisConfig, y: AxisConfig):
    ax.tick_params(direction='in', which='major', left=True, right=True, top=True, bottom=True, length=4, width=1.3)
    ax.tick_params(direction='in', which='minor', left=True, right=True, top=True, bottom=True, length=2, width=1)
    ax.spines[:].set_linewidth(1.3)
    ax.grid()
    x.apply(ax.xaxis, ax.set_xlabel)
    y.apply(ax.yaxis, ax.set_ylabel)
    
def make_text_box(text: str, x, y, ax):
    props = dict(facecolor='white', alpha=0.7)
    ax.text(x, y, text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', bbox=props)

def make_linegress_params(fit, precision_m, precision_b):
    m = fit.slope
    m_err = fit.stderr
    m_unit = 'mT/V'
    return f'$B_1 = {round(m, precision_m)} \pm {round(m_err, precision_m)} \; {m_unit}$\n$B_0 = {round(fit.intercept, precision_b)} \pm {round(fit.intercept_stderr, precision_b)} \;mT$'
    
def teil1():
    data = xw.Book('data/v30_messwerte.xlsx').sheets[0]
    X = data.range('D6:E28')
    U_spule = np.array(data.range('D6:D28').value)
    U_hall = np.array(data.range('E6:E28').value)

    fig, ax = plt.subplots()
    config_axis(ax, AxisConfig(label='$U_{Spule}$ in $V$', minors=AutoMinorLocator(5)), AxisConfig(label='$B_{Hall} $ in $mT$', minors=AutoMinorLocator(5)))
    #ax.plot(U_spule, U_hall, '.')

    B = 306.205 * U_hall**2 - 232.476 * U_hall - 0.669 # kalibrierungskurve
    ax.plot(U_spule, B, '.', color='red', label='B-Feld der Hall-Sonde')

    fit = stats.linregress(U_spule, B)
    ax.plot(U_spule, np.poly1d((fit.slope, fit.intercept))(U_spule), 'k--', label='Lin-Fit $B=B_1 \cdot U+B_0$')
    make_text_box(make_linegress_params(fit, 3, 3), 0.025, 0.82, ax)
    #print(fit)

    plt.legend()
    plt.savefig('tmp/v30_kalib.png')
    #plt.show()
    
def dicke(r1, r2):
    l = 643.85e-9
    n = 1.457
    f = 150.0e-3
    return l / (2*np.abs(np.sqrt(n**2 - np.sin(np.arctan(r2/f))**2) - np.sqrt(n**2 - np.sin(np.arctan(r1/f))**2)))

def teil2():
    r = np.array([7.5, 8.5, 9.2, 9.8]) - 5
    r /= 1000.0
    d = np.zeros(len(r)-1)
    for i in range(len(r)-1):
        d[i] = dicke(r[i], r[i+1]) * 1000
    print(d)
    print(np.mean(d))
    print(np.std(d))
    
#teil1()
teil2()
    