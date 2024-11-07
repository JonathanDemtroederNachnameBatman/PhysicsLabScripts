import matplotlib.pyplot as plt
import numpy as np


def lilienthal():
    x = [0.079, 0.034, 0.034, 0.0098, 0, 0, 0.049, 0.049, 0.049, 0.0393, 0.0589, 0.0737]
    y = [0.0982, 0.0295, 0, 0, 0, -0.0491, 0.0982, 0.0982, 0.1474, 0.1474, 0.1965, 0.2063]
    a = [0, 2, 4, 6, 8, 10, 12, 14, -2, -4, -6, -8]

    plt.grid()  
    plt.xlabel("cW")
    plt.ylabel("cA")
    #Splt.xticks(x)
    plt.scatter(x, y)
    plt.errorbar(x, y, yerr=)
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)

    for i in range(len(x)):
        if i == 6:
            plt.annotate(f'{a[i]}°, {a[i+1]}°', (x[i] + 0.001, y[i] + 0.002))
        elif i != 7:
            plt.annotate(f'{a[i]}°', (x[i] + 0.001, y[i] + 0.002))

    plt.show()

lilienthal()