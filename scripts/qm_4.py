import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(constrained_layout=True)

def trans(x, y):
    e = complex(x, y)
    U = 5.0
    L = 5.0
    k = np.sqrt(e)#np.sqrt(2*m*e) / h
    q = np.sqrt(e+U)#np.sqrt(2*m*e / h + U)
    s = np.abs(np.exp(-L*k*1j)/np.cos(L*q))**2 * np.abs(1/(1-(1j/2)*(k/q+q/k)*np.tan(L*q)))**2
    return s

n = 200
x = np.linspace(-4.75, -4.65, n)
y = np.linspace(-0.01, 0.01, n)

z = []

for b in y:
    z1 = []
    for a in x:
        #print(a + b*1j)
        z1.append(trans(a, b))
    z.append(z1)

z = np.array(z)

x, y = np.meshgrid(x, y)

fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot()

#ax.plot_surface(x, y, z)
cp = ax.contourf(x, y, z)
fig.colorbar(cp)

ax.set_xlabel('Re(ϵ)')
ax.set_ylabel('Img(ϵ)')
#ax.set_zlabel('|S(ϵ)|²')

plt.show()
