import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from etc import helpers as hp

def coulombwall():
    x = np.linspace(1, 5)
    y = 2 / (x)
    x = np.array([0, 1, 1, *x])
    y = np.array([-1, -1, 1.2, *y])
    
    fig, ax = plt.subplots(figsize=[10/2.54, 7.5/2.54])
    ax.config(hp.AxisConfig(ticks=False, label='Weg r', labelsize=10), hp.AxisConfig(ticks=False, label='Potential V', labelsize=10))
    ax.plot(x, y)
    ax.arrow(0.5, 1.1, 2, 0, head_width=0.08, head_length=0.15)
    plt.savefig('tmp/v25_coulombwall.png')
    plt.show()
    
def teil3():
    data = hp.excel('data/v25_messwerte.xlsx')
    ang = np.deg2rad(data.array('C41:C65'))
    n = data.array('D41:D65')
    eang = np.full(len(ang), np.deg2rad(2))
    en = np.sqrt(n)
    print(n)
    print(en)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.config(hp.AxisConfig(majors=np.pi/18, minors=2), hp.AxisConfig(label='Gemessene Zerfälle pro 60s'))
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.errorbar(ang, n, xerr=eang, yerr=en, fmt='r.', ecolor='black', capsize=2, markersize=5)
    plt.grid()
    plt.savefig('tmp/v25_polar.png')
    plt.show()
    
coulombwall()
#teil3()
