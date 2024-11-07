import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def errR(U, I, Ue, Ie):
    return np.sqrt((Ue/I)**2 + (U*Ie/(I*I))**2)

def teil2():
    X = pd.read_excel("data/v11_Messwerte.xlsx", usecols="D:G", skiprows=13, nrows=5, decimal=",", header=None)
    Ue = np.full(5, 0.001)
    Ie = np.array([0.01, 0.01, 0.001, 0.001, 0.001]) / 1000
    print("Spannungsrichtig")
    for i in range(5):
        print(f"Spann: {X[3][i] / (X[4][i] / 1000)} ± {errR(X[3][i], X[4][i] / 1000, Ue[i], Ie[i])}")
        print(f"Strom: {X[5][i] / (X[6][i] / 1000)} ± {errR(X[5][i], X[6][i] / 1000, Ue[i], Ie[i])}")

teil2()

def teil4():
    sperr_strom = pd.read_excel("data/v11_Messwerte.xlsx", usecols="C:D", skiprows=55, nrows=11, decimal=",", header=None)
    sperr_spann = pd.read_excel("data/v11_Messwerte.xlsx", usecols="E:F", skiprows=55, nrows=11, decimal=",", header=None)
    durch_strom = pd.read_excel("data/v11_Messwerte.xlsx", usecols="C:D", skiprows=70, nrows=11, decimal=",", header=None)
    durch_spann = pd.read_excel("data/v11_Messwerte.xlsx", usecols="E:F", skiprows=70, nrows=11, decimal=",", header=None)

    err_s = np.full(11, 0.001)
    err_m = np.full(11, 0.001)
    err_m[10] = 0.01
    err_l = [0.001, 0.001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    fig, ax = plt.subplots()

    if True:
        ax.errorbar(-sperr_strom[2], -sperr_strom[3], yerr=err_s, linewidth=0, elinewidth=1, capsize=5, marker='.', label="Stromrichtig / Sperrrichtung", alpha=0.75)
        ax.errorbar(-sperr_spann[4], -sperr_spann[5], yerr=err_s, linewidth=0, elinewidth=1, capsize=5, marker='.', label="Spannungsrichtig / Sperrrichtung", alpha=0.75)
        ax.errorbar(durch_strom[2], durch_strom[3], yerr=err_m, linewidth=0, elinewidth=1, capsize=5, marker='.', label="Stromrichtig / Durchlassrichtung", alpha=0.75)
        ax.errorbar(durch_spann[4], durch_spann[5], yerr=err_l, linewidth=0, elinewidth=1, capsize=5, marker='.', label="Spannungsrichtig / Durchlassrichtung", alpha=0.75)
    else:
        #ax.plot(durch_strom[2], durch_strom[3], linewidth=0, marker='.', label="Stromrichtig / Durchlassrichtung", alpha=0.75)
        #ax.plot(durch_spann[4], durch_spann[5], linewidth=0, marker='.', label="Spannungsrichtig / Durchlassrichtung", alpha=0.75)

        RiA = 1.0
        RiU = 1e6
        # stromrichtig
        Uk = durch_strom[2] - RiA * durch_strom[3] / 1000
        #spannungsrichtig
        Ik = ((durch_spann[5] / 1000) - durch_spann[4] / RiU) * 1000
        ax.errorbar(Uk, durch_strom[3], yerr=err_m, linewidth=0, elinewidth=1, capsize=5, marker='.', label="Stromrichtig / Spannung korrigiert", alpha=0.75)
        ax.errorbar(durch_spann[4], Ik, yerr=err_l, linewidth=0, elinewidth=1, capsize=5, marker='.', label="Spannungsrichtig / Strom korrigiert", alpha=0.75)
        
        print(np.abs(durch_strom[2] - Uk))
        print(np.abs(durch_spann[5] - Ik))
        print(np.mean(np.abs(durch_strom[2] - Uk)))
        print(np.mean(np.abs(durch_spann[5] - Ik)))

    ax.set_xlabel("U in V")
    ax.set_ylabel("I in mA")

    plt.legend()
    plt.grid()
    plt.show()
