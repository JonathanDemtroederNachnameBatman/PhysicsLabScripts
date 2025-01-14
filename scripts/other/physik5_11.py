import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from scipy import stats

from etc import helpers as hp

Re = 6378e3
g01 = -32e-6
h11 = 5e-6
g02 = -3.3e-6
h22 = -0.9e-6
r = np.array([2900e3, Re, 2*Re, 3*Re, 4*Re, 5*Re, 6*Re, 7*Re, 8*Re, 9*Re, 10*Re])
X = np.zeros(len(r))
Y = np.zeros(len(r))
Z = -2*(Re/r)**3*g01-3*(Re/r)**4*g02

fig, ax = plt.subplots()
ax.config(hp.AxisConfig(label='r in $R_E$'), hp.AxisConfig(label='Magnetfeld in $mT$'))
ax.set_title('Äquator')
ax.plot(r/Re, X*1e3, '.-', label='X')
ax.plot(r/Re, Y*1e3, '.-', label='Y')
ax.plot(r/Re, Z*1e3, '.-', label='Z')
ax.legend()
plt.savefig('../../tmp/physik5_11_aequator.png')
plt.show()

X = (Re/r)**3*g01
Z = -2*(Re/r)**3*g01-3*(Re/r)**4*g02

fig, ax = plt.subplots()
ax.config(hp.AxisConfig(label='r in $R_E$'), hp.AxisConfig(label='Magnetfeld in $mT$'))
ax.set_title('Pol')
ax.plot(r/Re, X*1e3, '.-', label='X')
#ax.plot(r/Re, Y, '.-', label='Y')
ax.plot(r/Re, Z*1e3, '.-', label='Z')
ax.legend()
plt.savefig('../../tmp/physik5_11_pol.png')
plt.show()
