import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from scipy import stats

#import etc.helpers as hp
from etc import helpers as hp


X = np.geomspace(1e-8, 1e-3, 1000)
#print(X)
h = np.longdouble(6.62607015e-34)
c = 299792458
k = np.longdouble(1.380649e-23)

#print(h*c/(k*X*300))

def plank(x, t):
    return 2*h*c*c / (x**5 * (np.exp(h*c/(k*x*t))-1))

fig, ax = plt.subplots()
ax.config(hp.AxisConfig(label='Wellenlänge $\\lambda$ in m', min=1e-8, max=1e-3), hp.AxisConfig(min=1e-2, max=1e20, label='spektrale Strahldicht $I_\\lambda$ in $Wm^{-3}sr^{-1}$'))
T = [300, 3600, 5800, 9900]
maxs = []
for t in T:
    I = plank(X, t)
    ax.plot(X, I, label=f'T={t} K')
    i = np.argmax(I)
    maxs.append([X[i], I[i]])
    ax.plot([X[i]], [I[i]], 'ok')
maxs = np.array(maxs).T
print(maxs[0])
maxs = np.log(maxs)
m = stats.linregress(maxs[0], maxs[1])
X = np.log([1e-8, 1e-3])
ax.plot(np.exp(X), np.exp(X * m.slope + m.intercept), 'k--', label='')
#ax.plot(maxs[0], maxs[1], 'k--')

ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.savefig('../../tmp/plank.png')
plt.show()
