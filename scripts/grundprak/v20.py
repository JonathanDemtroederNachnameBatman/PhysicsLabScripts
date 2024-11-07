import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R1 = 0.3
R2 = 0.5e6
C = 1e-6

l = np.divide([6.4, 11.6, 10.8, 37.5, 46.5, 52], 100)
n = [60, 110, 700, 250, 250, 500]
A = np.divide([0.37, 1.4, 1.8, 8.7, 14.2, 16], 10000)
names = ["Nano", "Amo", "Ferrit", "Mu", "Trafo", "Leybold"]

fig, ax = plt.subplots()

def avr_y(x, y):
    all0 = []
    for i in range(len(x)):
        if x[i] == 0 and y[i] > 0:
            all0.append(y[i])
    if len(all0) == 0:
        minI = -1
        maxI = -1
        for i in range(len(x)):
            if y[i] > 0:
                if x[i] < 0:
                    if minI < 0 or x[i] > x[minI]:
                        minI = i
                elif x[i] > 0:
                    if maxI < 0 or x[i] < x[maxI]:
                        maxI = i
        all0 = [y[minI], y[maxI]]
    return np.average(all0)

def avr_x(x, y):
    all0 = []
    for i in range(len(x)):
        if y[i] == 0 and x[i] < 0:
            all0.append(x[i])
    if len(all0) == 0:
        minI = -1
        maxI = -1
        for i in range(len(x)):
            if x[i] > 0:
                if y[i] < 0:
                    if minI < 0 or y[i] > y[minI]:
                        minI = i
                elif y[i] > 0:
                    if maxI < 0 or y[i] < y[maxI]:
                        maxI = i
        all0 = [x[minI], x[maxI]]
    return np.average(all0)

def diagram(i):
    X = pd.read_csv(f'data/v20/3_{names[i]}.txt', sep='\s+', decimal=',', skiprows=5, header=None)
    # print(X[0])

    Ux = X[1]
    Uy = X[2]
    H = Ux * (n[i] / (R1 * l[i]))
    B = Uy * (R2 * C / (n[i] * A[i]))

    maxB = abs(max(B, key=abs))
    maxUp = max(B)
    balancedMaxB = (maxB + maxUp) / 2
    remanenz = avr_y(H, B)
    koerz = abs(avr_x(H, B))
    print(f"{names[i]}: Max B = {round(maxB, 4)}, balanced Max B = {round(balancedMaxB, 4)}, remanenz = {round(remanenz, 4)}, koerz. = {round(koerz, 4)}")

    textstr = '\n'.join((
        r"$B_{S,schlecht} = %.2f$ T" % maxB,
        r"$B_{S} = %.2f$ T" % balancedMaxB,
        r"$B_{R} = %.2f$ T" % remanenz,
        r"$H_{C} = %.2f$ A/m" % koerz,
    ))

    ax.plot(H, B, marker="o", linewidth=0, markersize=1)
    ax.set_xlabel("H in A/m")
    ax.set_ylabel("B in T")
    ax.set_ylim(-maxB - 0.1 * maxB, maxB + 0.1 * maxB)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle='round', facecolor="white", alpha=0.5))
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    #plt.show()
    plt.savefig(f"tmp/3_{names[i]}.png")
    ax.clear()

#diagram(0)
#for i in range(0, 6):
#    diagram(i)

def find_tr(x, y):
    maxDistance = 0
    mx = 0
    my = 0
    for i in range(len(x)):
        if x[i] > 0 and y[i] > 0:
            dist = np.sqrt(x[i]**2 + y[i]**2)
            if dist > maxDistance:
                maxDistance = dist
                mx = x[i]
                my = y[i]
    return (mx, my)

def find_bl(x, y):
    maxDistance = 0
    mx = 0
    my = 0
    for i in range(len(x)):
        if x[i] < 0 and y[i] < 0:
            dist = np.sqrt(x[i]**2 + y[i]**2)
            if dist > maxDistance:
                maxDistance = dist
                mx = x[i]
                my = y[i]
    return (mx, my)

def perm(points):
    minP = -1
    for i in range(len(points)):
        p = points[i]
        if p[0] > 0:
            if minP < 0 or p[0] < points[minP][0]:
                minP = i
    return (points[minP][1] / points[minP][0]) * (1 / 1.256637e-6)


def kommutierung(i):
    maxB = 0
    scatters = []
    maxPoints = []
    for j in range(1, 4):
        X = pd.read_csv(f'data/v20/3_{names[i]}_{j}.txt', sep='\s+', decimal=',', skiprows=5, header=None)

        Ux = X[1]
        Uy = X[2]
        H = Ux * (n[i] / (R1 * l[i]))
        B = Uy * (R2 * C / (n[i] * A[i]))

        maxB = max(maxB, abs(max(B, key=abs)))
        
        maxPoints.append(find_tr(H, B))
        maxPoints.append(find_bl(H, B))

        scatters.append(plt.scatter(H, B, 1, alpha=0.1))
    textstr = '\n'.join((
        r'$\mu_r = %.6f$' % perm(maxPoints),
    ))
    ax.set_xlabel("H in A/m")
    ax.set_ylabel("B in T")
    ax.plot([j[0] for j in maxPoints], [j[1] for j in maxPoints], "ro")
    ax.set_ylim(-maxB - 0.1 * maxB, maxB + 0.1 * maxB)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle='round', facecolor="white", alpha=0.5))
    #plt.legend(scatters, ["1", "2", "3"])
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    #plt.show()
    plt.savefig(f"tmp/3_{names[i]}_komm.png")
    ax.clear()

kommutierung(3)
kommutierung(4)
kommutierung(5)
