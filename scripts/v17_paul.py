import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

def std(arr):
    r = 0
    n = len(arr)
    avr = np.average(arr)
    for i in arr:
        r += (i - avr)**2
    r = np.sqrt(r / (n*n - n))
    print(r)
    return r[0]


fig, ax = plt.subplots(constrained_layout=True)

x_end = 100
x = np.arange(0, x_end, 0.01)

plt.grid()
ax.set_xlabel('Gegenstandsweite g in cm')
ax.set_ylabel('Bildweite b in cm')

#secax = ax.secondary_xaxis('top')
#ax2 = ax.twinx()
ax.set_xlim((0, x_end))
#ax2.set_xlim((0, 100))
ax.xaxis.set_ticks(np.arange(0, x_end, 10))
#secax.xaxis.set_ticks(np.arange(5, 100, 10))
ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
ax.set_ylim((0, x_end))
#ax2.set_ylim((0, 100))
ax.set_yticks(np.arange(0, x_end, 10))
#ax2.set_yticks(np.arange(0, 100, 10))

##########################################
# HIER MESSWERTE FÜR g & b EINTRAGEN
# gr : Vergrößerung / kl: Verkleinerung jeweils in cm

p = 10
#p_gr = np.average([98.5, 98.5, 98.5, 98.5, 98.45]) # fern
#p_kl = np.average([45.5, 45.4, 45.6, 45.6, 45.4]) # fern

#p_gr = np.average([97.85, 97.9, 97.9, 97.8, 97.95]) # nah
#p_kl = np.average([45.55, 45.5, 45.6, 45.6, 45.4]) # nah

#p_gr = np.average([98.3, 98.3, 98.25, 98.3, 98.2]) # rot
#p_kl = np.average([19.7, 19.7, 19.8, 19.7, 19.8]) # rot

p_gr = np.average([98.3, 98.3, 98.2, 98.35, 98.3]) # blau
p_kl = np.average([20.8, 20.75, 20.7, 20.7, 20.77]) # blau

b_gr = p_gr - p
b_kl = p_kl - p

g_gr = 110 - p_gr + 0
g_kl = 110 - p_kl + 0

e = (g_gr - g_kl) * 10
a = (110 - p) * 10
f = (a**2-e**2) / (4*a) # bessel
df = np.sqrt((3*(a**2 + e**2)/(4*a**2))**2 + (1 * e/(2*a))**2)
print(f'{f} ± {df}')


##########################################

gr = -(b_gr / g_gr) * x + b_gr
vk = -(b_kl / g_kl) * x + b_kl

plt.plot(x, gr, '-', label="Vergrößerung")
plt.plot(x, vk, '-', label="Verkleinerung")

idx1 = np.argwhere(np.diff(np.sign(vk - gr))).flatten()
plt.plot(x[idx1], vk[idx1], 'r.')

idx2 = np.argwhere(np.diff(np.sign(vk - x))).flatten()
plt.plot(x[idx2], vk[idx2], 'r.')

idx3 = np.argwhere(np.diff(np.sign(gr - x))).flatten()
plt.plot(x[idx3], gr[idx3], 'r.', label='Schnittpunkte')

plt.plot(x, x, 'grey', linestyle='dotted')
plt.legend()

mx = round(np.average([x[idx1], x[idx2], x[idx3]]), 2)
my = round(np.average([vk[idx1], vk[idx2], gr[idx3]]), 2)
m = round((mx + my) / 2.0, 2)
dmx = round(std([x[idx1], x[idx2], x[idx3]]) + 0.1, 2)
dmy = round(std([vk[idx1], vk[idx2], gr[idx3]]) + 0.1, 2)
dm = round(np.sqrt((dmx/2)**2 + (dmy/2)**2), 2)

tx = 31
ty = 95
plt.text(tx, ty, "Mittelwert aller Schnittpunkte:")
plt.text(tx, ty - 5, f"in X:           {mx} ± {dmx} cm")
plt.text(tx, ty - 10, f"in Y:            {my} ± {dmy} cm")
plt.text(tx, ty - 15, f"in X und Y: {m} ± {dm} cm")

plt.show()

# Namen der Bilddatei anpassen!
# plt.savefig('plotVT4fern.png')
# plt.close()
