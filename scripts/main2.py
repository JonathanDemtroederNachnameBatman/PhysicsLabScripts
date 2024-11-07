
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


#Messdaten

x = [
84,
82,
80,
78,
76,
74,
72,
70,
68,
66,
64,
62,
60,
58,
15.5
]

y = [
0.00368,
0.00355,
0.00347,
0.00334,
0.00326,
0.00315,
0.00306,
0.00293,
0.00287,
0.00277,
0.00268,
0.00258,
0.00326,
0.0024,
0.0006
]

# Messdaten ohne Ausschlag

x2 = [
84,
82,
80,
78,
76,
74,
72,
70,
68,
66,
64,
62,

58,
15.5
]

y2 = [
0.00368,
0.00355,
0.00347,
0.00334,
0.00326,
0.00315,
0.00306,
0.00293,
0.00287,
0.00277,
0.00268,
0.00258,

0.0024,
0.0006
]

#Datenbereich Regressionsgerade 
x1 = np.linspace(10,90,100)

#Fehler Spannung und Temperatur
dy = 0.00005
dx = 0.5

# Ausgabe Messpunkte und Fehlerbalken
plt.errorbar(x, y, yerr=dy, xerr=dx, fmt='.',ecolor = 'black',color = 'blue', zorder=1)
plt.plot(x, y, '.', color = 'blue', label='Messdaten')

#Regression und Ausgabe
res = stats.linregress(x, y)
plt.plot(x1, res.intercept + res.slope*x1, 'red',linestyle='dashed', label='Lineare Regression')

#angepasste Regression und Ausgabe
res = stats.linregress(x2, y2)
plt.plot(x1, res.intercept + res.slope*x1, 'green',linestyle='dashed', label='Lineare Regression angepasst')


#Ausgabe Plotinformationen
plt.text(8,0.0031,"U = 4.47e-05 ΔT - 1.51e-04   ")
plt.text(8,0.0029,"R² = "+str(round(res.rvalue,4)))
plt.text(8,0.0027,"σ(C3)= 0.07e-05")
plt.text(8,0.0025,"σ(C4)= 0.47e-04")


plt.ylabel("U in V")
plt.xlabel("ΔT in K")
plt.legend()
plt.show()
