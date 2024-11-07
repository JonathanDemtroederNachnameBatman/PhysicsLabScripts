import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
#import lib.helpers as hp
from lib import helpers as hp

d = 57.3
U = np.pi * d
f = 360.0 / U

x = np.array([40.9, 58.8, 73.3, 87.5, 101.2, 115.3, 131.3, 153.1]) * f
x = np.deg2rad(x)
x0 = np.copy(x)
r = np.ones(len(x))

x = x / 4

print(np.rad2deg(x))

l = 1.54 * 1e-10

d = np.zeros(len(x))
for i in range(len(x)):
    d[i] = (i+1) * l / (2 * np.sin(x[i]))
    #d1 = (i+1) * l / (2 * np.sin(x1[i]))
    #d2 = (i+1) * l / (2 * np.sin(x2[i]))
    #print(f'd: {d1 * 1e12}pm   {d2 * 1e12}pm')
    #print(f'd: {d1 * 1e12}pm')
print(f'd: {d}')

fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
for i in range(len(x)):
    ax.plot([x0[i] / 4, 0, -x0[i] / 4], [1, 0, 1], label=f'$\\theta={round(np.rad2deg(x[i]), 2)}$')
#ax.plot([mid, mid + np.pi], [1, 1])

plt.legend()
plt.show()

"""
h:0 k:0 l:0, hkl:0.0
h:0 k:0 l:1, hkl:1.0
h:0 k:0 l:2, hkl:2.0
h:0 k:1 l:0, hkl:1.0
h:0 k:1 l:1, hkl:1.4142135623730951
h:0 k:1 l:2, hkl:2.23606797749979
h:0 k:2 l:0, hkl:2.0
h:0 k:2 l:1, hkl:2.23606797749979
h:0 k:2 l:2, hkl:2.8284271247461903
h:1 k:0 l:0, hkl:1.0
h:1 k:0 l:1, hkl:1.4142135623730951
h:1 k:0 l:2, hkl:2.23606797749979
h:1 k:1 l:0, hkl:1.4142135623730951
h:1 k:1 l:1, hkl:1.7320508075688772
h:1 k:1 l:2, hkl:2.449489742783178
h:1 k:2 l:0, hkl:2.23606797749979
h:1 k:2 l:1, hkl:2.449489742783178
h:1 k:2 l:2, hkl:3.0
h:2 k:0 l:0, hkl:2.0
h:2 k:0 l:1, hkl:2.23606797749979
h:2 k:0 l:2, hkl:2.8284271247461903
h:2 k:1 l:0, hkl:2.23606797749979
h:2 k:1 l:1, hkl:2.449489742783178
h:2 k:1 l:2, hkl:3.0
h:2 k:2 l:0, hkl:2.8284271247461903
h:2 k:2 l:1, hkl:3.0
h:2 k:2 l:2, hkl:3.4641016151377544

for h in range(3):
    for k in range(3):
        for l in range(3):
            a = np.sqrt(h*h + k*k + l*l)
            print(f'h:{h} k:{k} l:{l}, hkl:{a}')
"""
