import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

def make_text_box(text: str, x, y, ax):
    props = dict(facecolor='white', alpha=0.7)
    ax.text(x, y, text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=props)

def make_linegress_params(fit, precision_m, precision_b):
    m = fit.slope * 1e5
    m_err = fit.stderr * 1e5
    return f'm = ({round(m, precision_m)} ± {round(m_err, precision_m)})e-5\nb = {round(fit.intercept, precision_b)} ± {round(fit.intercept_stderr, precision_b)}'

x = [404.6, 407.8, 435.8, 491.6, 546.2, 579.06, 623.4]
#y = [1.4632, 1.4595, 1.4559, 1.4485, 1.4448, 1.4410, 1.4372]
y = [
    1.581,
    1.586,
    1.592,
    1.597,
    1.607,
    1.612,
    1.618
]
y.reverse()
err = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]

fit = stats.linregress(x, y)

f, ax = plt.subplots()

plt.plot(x, np.poly1d((fit.slope, fit.intercept))(x), '--k', label='Linearer fit')
plt.plot(x, y, 'o', label='n(λ)')
plt.xlabel('Wellenlänge λ')
plt.ylabel('Brechungsindex n(λ)')
plt.grid()
plt.legend()
plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
#plt.text(420,1.45, f'm = {round(fit.slope, 3)} ± {round(fit.stderr, 3)}')
#plt.text(420,1.455, f'b = {round(fit.intercept, 3)} ± {round(fit.intercept_stderr, 3)}')
make_text_box(make_linegress_params(fit, 3, 3), 0.975, 0.82, ax)

plt.show()
