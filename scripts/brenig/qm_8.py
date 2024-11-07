import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 1.

J = 200 # Ortsstützstellen
N = 200 # Zeitstützstellen
T = 1.0 # Endzeit
dx = 1.0 / J
dt = T / N

#x = np.arange(0.0, 1.0, dx)
#t = np.arange(0.0, 5.0, dt)

# Potential V(x) im bereich 0 - 1
def g(j):
    return 0#(j*dx)**2

# 2.

# Wellenfunktion von j bei t = 0
def ψ_j0(j):
    k = 30
    σ = 0.06
    x0 = 0.2
    return np.exp(1j * k * j*dx) * np.exp(-( (j*dx-x0)**2 ) / (2*σ**2))

# 2D liste aller Stützstellen der Wellenfunktion
ψ = [list(map(ψ_j0, range(0, J+1)))]
ψ[0][0] = 0   # Randbedingung
ψ[0][J] = 0

# Ω von j bei t = 0
def Ω_jn(j, n): # (22)
    a = 0 if j >= J else ψ[n][j+1] # ψ j+1
    b = 0 if j <= 0 else ψ[n][j-1] # ψ j-1
    return -a + (2j * (dx**2)/dt + 2 + dx**2 * g(j)) * ψ[n][j] - b

# 2D liste aller Stützstellen der Irgendwas mit Zeitentwicklung
Ω = [list(map(lambda j: Ω_jn(j, 0), range(0, J+1)))]

# 3.

# a_j mit Anfangswert 1
a = [0]
for j in range(1, J+1):
    d = 0 if j == 1 else 1 / a[j-1]
    a.append(2 + dx**2 * g(j) - 2j * (dx**2)/dt - d)

# 4.
    
def ψ_dx(j, n): # (17)
    ψ_p1 = 0 if j >= J else ψ[n][j+1]
    ψ_m1 = 0 if j <= 0 else ψ[n][j-1]
    return (ψ_p1 + ψ_m1 - 2 * ψ[n][j]) / (dx**2)

def ψ_jn(j, n): # (20)
    if j == 0 or j == J: # Randbedingung
        return 0
    d = ψ[n-1][j]
    H = -ψ_dx(j, n-1) + g(j) * d
    f = dt * 1j / 2
    return (d - f * H) / (d + f * H)

# Berechne alle psi und omega
for n in range(0, N+1): # für n = 1 bereits berechnet
    ψ.append(list(map(lambda j: ψ_jn(j, n), range(0, J+1))))
    Ω.append(list(map(lambda j: Ω_jn(j, n), range(0, J+1))))

b = []

print(len(a))
for n in range(0, N+1):
    b.append([])
    for j in range(0, J+1):
        d = 0 if j <= 1 else b[n][j-1] / a[j-1]
        b2 = Ω[n][j] + d
        b[n].append(b2)

# a und b werden berechnet aber sie werden nicht gebracuht?
# Ω ebenfalls

x = np.arange(0, 1, dx)

fig, ax = plt.subplots()
ax.set(xlim=[0, 1], ylim = [-0, 10], xlabel="x", ylabel="Wahrscheinlichkeit")

#x = np.arange(0, 10, 0.1)
#print(x)

line = ax.plot([], [])[0]

startT = int(0)

def update(n):
    y = []
    for j in range(1, J+1):
        #print(a + b*1j)
        y.append(np.abs(ψ[n - startT][j])**2)
    #print(y)
    #ax.plot(x, y)
    line.set_xdata(x)
    line.set_ydata(y)
    ax.set_title(f"t = {'{:.3f}'.format(np.round((n+startT) * dt, 3))}")
    return line,

#update(50)

ani = animation.FuncAnimation(fig=fig, func=update, frames=(N - startT), interval=40, repeat=True)
plt.grid()
#plt.show()

ani.save(filename="tmp/qm_8.gif", writer="pillow")
