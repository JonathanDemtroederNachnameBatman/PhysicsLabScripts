import random
import numpy as np
import matplotlib.pyplot as plt

# a)

def simple_monte_carlo(N):
    # Integrationsgrenzen
    x_min = 1.0
    x_max = 2.0
   
    integral = 0.0
    for _ in range(0, N):
        # Zufälliger Punkt inerhalb der Grenzen
        x = random.uniform(x_min, x_max)
        # y Wert berechnen
        y = np.sqrt(x**2 - 1.0)
        integral += y
    # Formel anwenden
    return ((x_max - x_min) / float(N)) * integral * 2

def a():
    # Anzahl Zufallspunkte
    x = [10, 100, 1000, 10000, 100000, 1000000]
    y = []
    # Per Hand berechneter Wert
    excepted = 2.147143718
    for N in x:
        errors = []
        # 5 durchläufe und Fehler dann mitteln
        for _ in range(0, 5):    
            area = simple_monte_carlo(N)
            errors.append(np.abs(excepted - area) / excepted)
        y.append(np.mean(errors))  

    plt.xlabel("N")
    plt.ylabel("Fehler")
    plt.xticks(x)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x, y)
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    plt.show()

a()
print(simple_monte_carlo(10000))


# b)

N = 100000
# Integrationsgrenzen
x_min = 0
x_max = 1
y_min = 0
y_max = 1

integral = 0.0
for i in range(0, N):
    # Zufälliger Punkt
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    # z berechnen
    z = 1 / (1 + (x**2)*(y**2))
    integral += z
    
# Volumen berechnen
volume = (x_max - x_min) * (y_max - y_min) / float(N) * integral
print(volume)
# 3 berechnete Volumina
# 0.9163571626528748
# 0.9158606946063141
# 0.9160000487910218 
