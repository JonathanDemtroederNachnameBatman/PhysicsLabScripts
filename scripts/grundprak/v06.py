import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

def parse_list(lst):
    values = list(lst)
    for i in range(len(values)):
        if isinstance(values[i], str):
            values[i] = float(values[i][:-2])
        values[i] = round(values[i], 1) 
    return values

def from_excel(path, startRow, endRow, cols, name):
    sheet = pd.read_excel(path, 'Tabelle1', skiprows = startRow-1, nrows = endRow-startRow, usecols=cols, dtype=np.float64)
    #print(sheet[name])
    return parse_list(sheet[name])

def teil_1():
    y = from_excel('data/v06_messwerte.xlsx', 3, 26, 'B', 'Druck in mbar')
    x = from_excel('data/v06_messwerte.xlsx', 3, 26, 'C', 'Temp in C')

    print(x)
    print(y)

    err = []
    xerr = []

    for i in range(len(y)):
        err.append(2)
        xerr.append(0.5)
        y[i] = 1003 + y[i]

    x2 = x[18:-1]
    y2 = y[18:-1]
    err2 = err[18:-1]
    xerr2 = xerr[18:-1]
    x1 = x[1:19]
    y1 = y[1:19]
    err1 = err[1:19]
    xerr1 = xerr[1:19]

    fit = stats.linregress(x, y)
    print(fit)

    plt.plot(x, np.poly1d((fit.slope, fit.intercept))(x), '--k', label='Linearer fit')
    #plt.plot(x2, np.poly1d(fit2)(x2), '--k', label='Abkühlen fit')
    plt.errorbar(x1, y1, err1, xerr1, fmt='.', label='Erhitzen', capsize=1.5)
    plt.errorbar(x2, y2, err2, xerr2, fmt='.', label='Abkühlen', capsize=1.5)
    plt.ylabel('Druck in mbar')
    plt.xlabel('Temp in °C')
    plt.grid()
    plt.legend()
    plt.text(19,1143, f'm = {round(fit.slope, 3)} ± {round(fit.stderr, 3)}')
    plt.text(19,1136, f'b = {round(fit.intercept, 3)} ± {round(fit.intercept_stderr, 3)}')

    plt.show()


def teil_2():
    y = from_excel('data/v06_messwerte.xlsx', 4, 16, 'F', 'Druck bar?')
    x = from_excel('data/v06_messwerte.xlsx', 4, 16, 'G', 'Temp in C')
    print(y)
    print(x)

    yerr = []
    xerr = []

    for i in range(len(y)):
        #yerr.append(100000 * 0.01)
        #xerr.append(1.0/274.15)
        y[i] = 100300 - 100000 * (1-y[i])
        y[i] = np.log(y[i])
        x[i] = 273.15 + x[i]
        x[i] = 1.0/x[i]
        yerr.append(0.02 * 100000.0 / ((100000.0 * y[i] + 300)))
        xerr.append(1 * 1.0 / (x[i] + 273.15)**2)

    fit = stats.linregress(x[2:-1], y[2:-1])
    print(fit)

    plt.plot(x, np.poly1d((fit.slope, fit.intercept))(x), '--k', label='Linearer fit')
    plt.errorbar(x, y, yerr, xerr, fmt='.', label='Dampfkurve', capsize=1.5)
    plt.ylabel('Druck in ln(Pa)')
    plt.xlabel('Temp in 1/K')
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    plt.legend()
    plt.text(0.00272,9.02, f'm = {round(fit.slope, 3)} ± {round(fit.stderr, 3)}')
    plt.text(0.00272,8.82, f'b = {round(fit.intercept, 3)} ± {round(fit.intercept_stderr, 3)}')

    plt.show()


teil_1()