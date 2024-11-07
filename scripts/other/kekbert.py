
import math
import numpy as np
import  matplotlib.pyplot as plt

x = np.ones(100)
a = 3.5
x[0] = 0.12345

for k in range(len(x)-1):
    x[k+1] = a * x[k] * (1 - x[k])

plt.plot(x, '.-')
plt.grid()
plt.show()

