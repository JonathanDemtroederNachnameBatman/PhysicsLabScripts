import matplotlib.pyplot as plt
import numpy as np

def log(values):
    for i in range(0, len(values)):
        values[i] = np.log10(values[i])
    return values

def daempfung_log():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = log([10, 9.5, 9, 8.6, 8.2, 7.8, 7.2, 6.9, 6.6, 6.2, 6])
    y2 = log([10, 9, 8.3, 7.6, 7, 6.5, 5.9, 5.5, 5, 4.6, 4.2])
    y3 = log([10, 8.7, 7.5, 6.5, 5.6, 4.9, 4.2, 3.7, 3.2, 2.7, 2.3])
    y4 = log([10, 7.9, 6.4, 5.1, 4.1, 3.2, 2.5, 1.9, 1.4, 1.2, 0.8])
    y5 = log([10, 7.3, 5.3, 3.9, 2.8, 1.9, 1.4, 0.9, 0.7, 0.4, 0.3])

    #plt.errorbar(x, y, e, linestyle='None', marker='.', capsize=4)
    #plt.title(label="Messwerte gedämpfte Schwingung - 0,1 A")
    plt.xlabel("Periode k")
    plt.ylabel("Amplitude")
    plt.plot(x, y, label="0,1 A")
    plt.plot(x, y2, label="0,2 A") 
    plt.plot(x, y3, label="0,3 A") 
    plt.plot(x, y4, label="0,4 A") 
    plt.plot(x, y5, label="0,5 A") 
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    #plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()

def daempfung_T():
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    y1 = [1.8, 1.809, 1.806, 1.798, 1.786]

    plt.xlabel("Stromstärke I [A]")
    plt.ylabel("Schwingungsdauer T [s]")
    plt.xticks(x)
    plt.plot(x, y1, 'o', label='Schwingungsdauer')
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    plt.show()

def daempfung_lambda():
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    y2 = [0.051, 0.087, 0.147, 0.253, 0.354]

    plt.xlabel("Stromstärke I [A]")
    plt.ylabel("log. Dekrement λ")
    plt.xticks(x)
    plt.plot(x, y2, 'or', label='log. Dekrement')
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    plt.show()

def linear_regression(x, y):
    coef = np.polyfit(x, y, 1)
    print(coef)
    f = np.poly1d(coef)
    plt.plot(x, f(x), '--k')

def traegheit_der_anzeige():
    messwerte = [
        [0, 0, 0],
        [2.51, 3.10, 2.69],
        [6.66, 6.38, 5.55],
        [8.01, 10.12, 9.08],
        [9.99, 11.55, 10.34],
        [10.92, 13.52, 12.82],
        [12.05, 15.04, 14.43],
        [13.63, 17.14, 15.60],
        [15.92, 19.77, 19.58],
        [20.38, 23.98, 23.69],
        [24.34, 29.02, 27.01],
        [30.87, 36.50, 34.90],
        [45.42, 51.00, 47.62],
        [63.61, 76.96, 73.58]
    ]
    x = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25]
    x = [c - 25 for c in x]
    #x = [70, 65, 60, 55, 50, 45, 40, 35, 30, 25]
    ticks = [90, 75, 65, 55, 45, 40, 35, 30, 25]
    ticks = [c - 25 for c in ticks]
    #ticks = [70, 65, 60, 55, 50, 45, 40, 35, 30, 25]
    y = [0, 2.51, 6.66, 8.01, 9.99, 10.92, 12.05, 13.63, 15.92, 20.83, 24.34, 30.87, 45.42, 63.61]
    mean = [(np.mean(data)) for data in messwerte]
    error = [np.std(data) for data in messwerte]
    y2 = log([10, 9, 8.3, 7.6, 7, 6.5, 5.9, 5.5, 5, 4.6, 4.2])
    y3 = log([10, 8.7, 7.5, 6.5, 5.6, 4.9, 4.2, 3.7, 3.2, 2.7, 2.3])

    x = [np.log(c) if c > 0 else c for c in x]

    #plt.errorbar(x, y, e, linestyle='None', marker='.', capsize=4)
    #plt.title(label="Messwerte gedämpfte Schwingung - 0,1 A")
    plt.ylabel("Temperatur differenz (T- T0) °C")
    plt.xlabel("Zeit t")
    plt.yticks(ticks=[np.log(c) if c > 0 else c for c in ticks], labels=[str(label) for label in ticks])
    #plt.errorbar(x, mean, error, fmt='.', capsize=3)
    plt.errorbar(mean, x, xerr=error, fmt='.', capsize=3, label="Alkoholthermometer")
    linear_regression(mean, x)
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    #plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()

def widerstand_zu_celsius(ohm):
    return (4265.22/(np.log(ohm * 1000) + 3.62))

def spannung_zu_celsius(u):
    return ((u / 1000 + 0.000151) / 0.0000447)

def oberflächentemperatur():
    x = [61, 60, 58, 56, 54, 51, 50, 48, 46, 44, 42, 40]
    x = [c + 273.15 for c in x]
    y1 = [19.4, 21.72, 23.33, 24.9, 27.45, 28, 28.15, 26.55, 27.4, 28.4, 29.8, 31.6]
    y2 = [1.46, 1.39, 1.43, 1.40, 1.43, 1.31, 1.3, 1.29, 1.24, 1.19, 1.17, 1.15]
    error_widerstand = []
    error_thermo = []
    for i in range(0, len(y1)):
        error_widerstand.append(5.5)
        error_thermo.append(1.9)
    #x.reverse()
    y1 = [widerstand_zu_celsius(ohm) for ohm in y1]
    y2 = [spannung_zu_celsius(u) + 273.15 for u in y2]
    #y1 = log(y1)
    plt.xlabel("Temperatur K mit Alkoholthermometer")
    plt.ylabel("Temperatur K umgerechnet")
    plt.errorbar(x, y1, error_widerstand, linestyle='None', marker='.', capsize=4, label="Widerstand")
    plt.errorbar(x, y2, error_thermo, linestyle='None', marker='.', capsize=4, label="Thermoelement")
    #plt.plot(x, y1, 'o', label="Widerstand")
    #plt.plot(x, y2, 'o', label="Thermoelement")
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    #plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()

traegheit_der_anzeige()
#oberflächentemperatur()


