import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from etc import helpers as hp

def C_G_func(U, C_R, C_0, U_D):
    return C_R + C_0 * np.sqrt(U_D / (U_D - U))

def teil1():
    data = hp.excel('data/v21_messwerte.xlsx')
    Ux = data.array('C82:C106')
    f = data.array('D82:D106') * 1e3
    ef = 100
    L = 0.01
    C_G = 1 / (L*(2*np.pi*f)**2)
    eC_G = ef / (L * 2*np.pi**2 * f**3)
    #print(eC_G)
    popt, pcov = optimize.curve_fit(C_G_func, Ux, C_G)
    C_R, C_0, U_D = popt
    eC_R, eC_0, eU_D = np.sqrt(np.diag(pcov))
    Ux2 = np.linspace(0, -12.0, 100)
    C_S = C_G_func(Ux, *popt) - C_R
    eC_S = np.sqrt(eC_G**2 + eC_R**2)
    print(C_G_func(-4, *popt) * 1e12)
    p = '\\cdot10^{-12}'
    scale = 1e12
    presc = 1
    text = f'$C_R=({round(C_R * scale, presc)}\pm{round(eC_R * scale, presc)}){p}F$\n$C_0=({round(C_0 * scale, presc)}\pm{round(eC_0 * scale, presc)}){p}F$\n$U_D=({round(U_D, 3)}\pm{round(eU_D, 3)})V$'
    
    fig, ax = plt.subplots()
    if True:
        ax.config(hp.AxisConfig(minors=2, label='$-U$ in $V$'), hp.AxisConfig(minors=2, label='$C$ in $nF$'))
        
        ax.plot(Ux2, C_G_func(Ux2, *popt) * 1e9, 'k--', label='Nicht-linearer fit $C_R+C_O\\sqrt{U_D/(U_D - U)}$')
        ax.errorbar(Ux, C_S * 1e9, yerr=eC_S*1e9, fmt='b.', label='$C_S=C_G-C_R$', capsize=5)
        ax.errorbar(Ux, C_G * 1e9, yerr=eC_G*1e9, fmt='r.', label='Gesamtkapazität $C_G$', capsize=5)
        ax.text_box(text, 0.025, 0.73)
        ax.legend()
        plt.savefig('tmp/cg.png')
    else:
        d = 4.7e-3
        ed = 0.05e-3
        A = d**2 * np.pi / 4
        eps = 11.7 * 8.8541878128e-12
        B = A * eps
        d_S = B / C_S
        ed_S = np.pi * d * eps * np.sqrt((2*ed/C_S)**2 + ((d * eC_S) / (C_S**2))**2)
        d_0 = B / C_0
        ed0 = np.pi * d * eps * np.sqrt((2*ed/C_0)**2 + ((d * eC_0) / (C_0**2))**2)
        ax.config(hp.AxisConfig(minors=2, label='$-U$ in $V$'), hp.AxisConfig(minors=5, label='$d$ in $\mu m$'))
        ax.errorbar(Ux, d_S * 1e6, yerr=ed_S * 1e6, fmt='r.', label='$d_S(U)$', capsize=5)
        ax.errorbar([0], [d_0 * 1e6], yerr=ed0*1e6, fmt='b.', label=f'$d_0=({round(d_0 * 1e6, 3)}\pm{round(ed0 * 1e6, 3)})\\mu m$', capsize=5)
        
        ax.legend()
        plt.savefig('tmp/ds.png')
    
    plt.show()
    
teil1()