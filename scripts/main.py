import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

print('Hello World')

"""
def f(x):
    return (x-2) ** 3 - x ** 2 + 10

def gen_graph(function, min, max):
    xValues = []
    arr = []
    i = (max - min) / 100
    x = min
    while x < max:
        xValues.append(x)
        arr.append(function(x))
        x += i
    plt.plot(xValues, arr)


# gen_graph(f, -1, 6)

u1 = [1.18, 2.25, 5.07, 6.9, 7.95]
i1 = [0.0066, 0.0127, 0.0287, 0.0391, 0.045]

u2 = [1, 2, 3, 4.03, 5.04, 5.97]
i2 = [0.1, 0.15, 0.19, 0.224, 0.255, 0.279]

#plt.plot(u1, i1, label='1. Widerstand')
plt.plot(u2, i2, label='Lampe')
plt.ylabel('Stromstärke I (A)')
plt.xlabel('Spannung U (V)')
plt.grid()
plt.legend()

plt.show()
"""

fig, ax = plt.subplots()
ax.set(xlim=[0, 10], ylim = [-0.1, 1.3])

x = np.arange(0, 10, 0.1)
#print(x)

line = ax.plot([], [])[0]

def update(frame):
    y = np.exp(-(x-frame/10)**2)
    #print(y)
    #ax.plot(x, y)
    line.set_xdata(x)
    line.set_ydata(y)
    return line,

#update(50)

ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=10)
plt.grid()
plt.show()
