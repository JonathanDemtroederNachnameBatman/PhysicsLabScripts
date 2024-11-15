from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, Locator)
from matplotlib.axes import Axes
import numbers
import xlwings as xw
import numpy as np
from scipy import optimize

"""Beipiel
from lib import helpers as hp  # geht von lib/helpers.py aus

data = hp.excel('data/blabla.xlsx')  # Das erste Blatt einer excel datei

x = data.array('C5:C20')  # Gibt ein numpy array mit den angegebenen zellen

fig, ax = plt.subplots()
ax.config(hp.AxisConfig(label='$U_0$'), hp.AxisConfig())  # standard Achsen Konfiguration (siehe unten für parameter)

ax.text_box('Text', 0.025, 0.8)  # Fügt eine textbox mit relativer positionierung (0-1) von links und oben  
"""

# helper class to easily decorate multiple plots with similar data
class AxisConfig:

    def __init__(self, min=None, max=None, minors=None, majors=None, label=None, ticks=True, labelsize=10):
        self.min = min
        self.max = max
        self.minors = minors
        self.majors = majors
        self.label = label
        self.ticks = ticks
        self.labelsize = labelsize

    def apply(self, axis, set_label, set_lim):
        if self.majors: 
            if isinstance(self.majors, numbers.Number):
                axis.set_major_locator(MultipleLocator(self.majors))
            elif isinstance(self.majors, Locator):
                axis.set_major_locator(self.majors)
            else: 
                raise Exception('Wrong majors type')
        if self.minors: 
            if isinstance(self.minors, numbers.Number):
                axis.set_minor_locator(AutoMinorLocator(self.minors))
            elif isinstance(self.minors, Locator):
                axis.set_minor_locator(self.minors)
            else: 
                raise Exception('Wrong minors type')
        if self.label: set_label(self.label, fontsize=self.labelsize, fontweight='medium')
        if not self.ticks: axis.set_ticklabels([])
        if self.min or self.max: set_lim(self.min, self.max)

# helper method to easily decorate a plot
def config_axis(self, x=None, y=None):
    self.tick_params(direction='in', which='major', left=True, right=True, top=True, bottom=True, length=4, width=1.3)
    self.tick_params(direction='in', which='minor', left=True, right=True, top=True, bottom=True, length=2, width=1)
    self.spines[:].set_linewidth(1.3)
    self.grid()
    if x: x.apply(self.xaxis, self.set_xlabel, self.set_xlim)
    if y: y.apply(self.yaxis, self.set_ylabel, self.set_ylim)
    
def config_xaxis(self, x):
    self.config(x)
    
def config_yaxis(self, y):
    self.config(y=y)
    
def make_text_box(self, text: str, x, y, vertical='top', horizontal='left', props=dict(facecolor='white', alpha=0.7)):
    self.text(x, y, text, transform=self.transAxes, verticalalignment=vertical, horizontalalignment=horizontal, bbox=props)

def excel(path: str):
    return xw.Book(path).sheets[0]

def col(self, range: str):
    return np.array(self.range(range).value)

def single_param_text(name, value, error, precision):
    return f'${name}={round(value, precision)}\\pm{round(error, precision)}$'

def param_text(param_tuples):
    # param tuple: (name, value, error, precision)
    text = ''
    for param in param_tuples:
        text += single_param_text(param[0], param[1], param[2], param[3]) + '\n'
    return text

def param_text_fit(names, popt, pcov, precision):
    precision = np.full(len(names), precision) if isinstance(precision, numbers.Number) else precision
    perr = np.sqrt(np.diag(pcov))
    l = min(len(names), len(popt), len(perr))
    text = ''
    for i in range(l):
        text += single_param_text(names[i], popt[i], perr[i], precision[i]) + '\n'
    return text

def curve_fit(func, xdata, ydata, *args, **kwargs):
    popt, pcov = optimize.curve_fit(func, xdata, ydata, args, kwargs)
    res = ydata - func(xdata, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    r = 1 - (ss_res / ss_tot)
    return popt, pcov, r

Axes.config = config_axis
Axes.configx = config_xaxis
Axes.configy = config_yaxis
Axes.text_box = make_text_box
xw.Sheet.array = col
