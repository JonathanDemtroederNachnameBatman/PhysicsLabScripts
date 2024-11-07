import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from etc import helpers as hp

d = 1.0/60.0
x = np.array([[241+50*d, 280.0], [246+50*d, 275.0], [248+20*d, 273+20*d]])
x = np.deg2rad(x)
n = 1
g = 1/500.0 * 1e-3
h = 6.626
c = 2.99792458
e = 1.602

l = np.zeros(3)
le = np.zeros(3)
R = np.zeros(3)
Re = np.zeros(3)
for i in range(len(x)):
    alpha = (x[i][1]-x[i][0]) / 2.0
    l[i] = g * np.sin(alpha) / n
    le[i] = g * np.cos(alpha) * 5*d * np.sqrt(2) / 2
    R[i] = h*c / (l[i] * (0.25 - 1/((i+3)**2))) / e * 1e-7
    Re[i] = h*c / (l[i]**2 * (0.25 - 1/((i+3)**2))) * le[i] / e * 1e-7
    print(f'λ{i+1}: ${round(l[i]*1e9, 2)}\pm{round(le[i]*1e9, 2)}$')
    print(f'R_H{i+1}: ${round(R[i], 3)}\pm{round(Re[i], 3)}$')
    