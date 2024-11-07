import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import pandas as pd
from scipy import stats

def sgn(x):
    # Warum hat python keine signum funktion?
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

# Bestimmt nullstlle um index durch lineare regression
def find_zero_at(x, y, i):
    s = 15
    xm = x[i-s:i+s]
    ym = y[i-s:i+s]
    model = stats.linregress(xm, ym)
    zero = -model.intercept / model.slope
    j = min(range(len(xm)), key=lambda k: abs(xm[k]-zero)) + i-s
    #print(f"Found exact zero at {np.round(zero, 3)} with closest {np.round(x[j], 3)}")
    return zero

# findet alle nullstellen durch suchen von Bereichen um 0
def find_zeros(x, y):
    exact_zeros = []
    # Es befindet sich höchstens eine Nullstelle in dieser Anzahl aufeinander folgender Werte
    nullstellen_aufloesung = 15
    zeros = []
    last_zero = -1
    for i in range(100, len(x)-100):
        if y[i] == 0 or sgn(y[i]) != sgn(y[i-1]):
            zeros.append(i)
            last_zero = i
        elif last_zero >= 0 and i - last_zero > nullstellen_aufloesung:
            #print(f"Found zero! {len(zeros)} indices and average x of {np.average([x[j] for j in zeros])} : {zeros}")
            exact_zeros.append(find_zero_at(x, y, int(np.average(zeros))))
            zeros = []
            last_zero = -1
    return np.array(exact_zeros)

def plot_funcs(x, y1, y2):
    fig, ax = plt.subplots()
    ax.plot(x, y1, marker=".", linewidth=0, markersize=1)
    ax.plot(x, y2, marker=".", linewidth=0, markersize=1)
    #ax.plot(zeros, np.zeros(len(zeros)), marker=".", linewidth=0, color="green")
    ax.set_xlabel("t in s")
    ax.set_ylabel("U in V")
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    plt.show()

def fft_test():
    X = pd.read_csv(f'data/v13/{1}_{int(200)}.txt', sep='\s+', decimal='.', skiprows=3, header=None)
    x = np.array(X[0])
    unit = np.average(np.diff(x))
    y1 = np.array(X[1])
    y2 = np.array(X[2])
    f = np.fft.fft(y1)
    #plt.plot(x, y1, '.')
    freq = np.fft.fftfreq(len(x)) / unit
    fig, ax = plt.subplots()
    ax.set_xlim(-0.3, 0.3)
    #trans = mtrans.Affine2D().scale(1/unit, 1)
    ax.plot(freq, np.abs(f), '.')
    print(freq[np.argmax(np.abs(f))] / unit)
    ax.grid()
    plt.show()

def ani_test():
    print(anim.writers.list())
    x = np.linspace(0, 6, 400)

    s1 = np.sin(x * ((np.pi * 2)))
    s2 = np.sin(x * ((np.pi * 2)) * 1.333)
    s3 = np.sin(x * ((np.pi * 2)) * 1.875)
    s = s1 + s2 + s3

    fig, ax = plt.subplots()
    ax.set_xlim(-.5, 6.5)
    ax.set_ylim(-1.1, 1.1)
    line = ax.plot([], [], '.')[0]
    def update(frame):
        line.set_xdata(x[:frame])
        line.set_ydata(s1[:frame])
        print(f'Frame {frame}   ', end='\r')
        return line,

    an = anim.FuncAnimation(fig=fig, func=update, frames=len(x), interval=25, repeat=True, cache_frame_data=False)

    #ax.plot(x, s1, '.', label='Signal 1', alpha=0.3)
    #plt.plot(x, s2, '.', label='Signal 2', alpha=0.3)
    #plt.plot(x, s3, '.', label='Signal 3', alpha=0.3)
    #plt.plot(x, s, '.', label='Signal')

    #plt.legend()
    ax.grid()
    plt.show()

ani_test()

def phasenverschiebung(prefix, freq):
    X = pd.read_csv(f'data/v13/{prefix}_{int(freq)}.txt', sep='\s+', decimal='.', skiprows=3, header=None)
    x = np.array(X[0])
    # Glättung der y durch "Fourier"-Faltung
    box_pts = 100
    box = np.ones(box_pts) / box_pts
    y1 = np.convolve(np.array(X[1]), box, mode="same")
    y2 = np.convolve(np.array(X[2]), box, mode="same")
    # Y korrektur
    yd = np.abs(np.max(y1)) - np.abs(np.min(y1))
    y1 = y1 - (yd/2)
    yd = np.abs(np.max(y2)) - np.abs(np.min(y2))
    y2 = y2 - (yd/2)
    # Nullstellen finden
    zeros1 = find_zeros(x, y1)
    zeros2 = find_zeros(x, y2)
    # Differenz aller Nullstellen
    # kompliziert wenn unterschiedlich viele Nullstellen
    if len(zeros1) != len(zeros2):
        if len(zeros1) > len(zeros2):
            l = zeros1
            s = zeros2
        else:
            l = zeros2
            s = zeros1
        t1 = l[:len(s)]
        t2 = l[len(l)-len(s):]
        d1 = np.average(np.diff(t1 - s))
        d2 = np.average(np.diff(t2 - s))
        if d2 > d1:
            diff = t1 - s
        else:
            diff = t2 - s
    else:     
        diff = zeros2 - zeros1;
    # Phasenverschiebung
    phi = 2*np.pi * (np.average(diff)/1000) / (1/freq)
    phiErr = 2*np.pi * (np.std(diff)/1000) / (1/freq)
    if phi < 0:
        phi += np.pi
    #print(diff)
    print(f"φ of f={freq} is {phi}")
    c = 30000
    #plot_funcs(x[:c], y1[:c], y2[:c])
    return phi, phiErr

def teil1():
    phi = []
    phiErr = []
    omega = []
    for f in range(1, 11):
        freq = f * 100
        omega.append(freq * 2 * np.pi)
        p, pe = phasenverschiebung("1", freq)
        phi.append(p)
        phiErr.append(pe)

    omegath = np.arange(90, 1010, 1) * np.pi * 2
    phith = np.arctan((1/(omegath*150*4.3e-6)))

    fig, ax = plt.subplots()
    if True:
        ax.errorbar(omega, phi, phiErr, marker=".", linewidth=0, markersize=7, elinewidth=1, capsize=4, label="Gemessen")
        ax.plot(omegath, phith, label="Theoretsich")
        ax.set_xlabel("Kreisfrequenz $\omega$ in rad/s")
        ax.set_ylabel("Phasenverschiebung $\\varphi$ in rad")
    else:
        n = 5
        Rc = -np.tan(phi[n]) * 150
        plt.arrow(0, 0, 150, Rc, length_includes_head=True, head_width=3, head_length=10)
        ax.set_ylabel("Im(Z)")
        ax.set_xlabel("Re(Z)")
        textstr = '\n'.join((
            r"Re(Z) = 150 $\Omega$",
            r"Im(Z) = %.2f $\Omega$" % Rc,
            r"$\varphi = %.2f$°" % np.rad2deg(phi[n]),
            r"$\omega = %.2f$ rad/s" % omega[n],
        ))
        ax.text(0.85, 0.95, textstr, transform=ax.transAxes, verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle='round', facecolor="white", alpha=0.5))
        ax.plot([150, 150], [0, Rc], '--k')
        ax.plot([0, 150], [Rc, Rc], '--k')
        ax.set_ylim(-75, 0)
        ax.set_xlim(0, 160)
        #print(f"Phi {phi[n]}, Rc {Rc}, ω {omega[n]}, f {omega[n] / (2*np.pi)}")
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    #plt.legend()
    plt.show()

def teil2():
    phi1 = []
    phiErr1 = []
    phi2 = []
    phiErr2 = []
    omega = []
    for f in range(1, 11):
        freq = f * 100
        omega.append(freq * 2 * np.pi)
        p, pe = phasenverschiebung("2a", freq);
        phi1.append(np.pi - p)
        phiErr1.append(pe)
        p, pe = phasenverschiebung("2b", freq);
        phi2.append(np.pi - p)
        phiErr2.append(pe)
    fig, ax = plt.subplots()

    fit = stats.linregress(omega, phi2)
    phi2_b = np.poly1d((fit.slope, fit.intercept))(omega)
    
    if False:
        # Plotte phi
        ax.errorbar(omega, phi1, phiErr1, marker=".", linewidth=0, markersize=7, elinewidth=1, capsize=4, label="Ohne Kern")
        ax.errorbar(omega, phi2, phiErr2, marker=".", linewidth=0, markersize=7, elinewidth=1, capsize=4, label="Mit Kern")
        #ax.plot(omega, phi2_b, '--k', label='Linearer fit')
        ax.set_ylabel("Phasenverschiebung $\\varphi$ in rad")
    else:
        # Plotte L
        omega = np.array(omega)
        phi1 = np.array(phi1)
        phi2 = np.array(phi2)
        phiErr1 = np.array(phiErr1)
        phiErr2 = np.array(phiErr2)

        L2 = (np.tan(phi2) * 152.5) / omega
        L1 = (np.tan(phi1) * 152.5) / omega
        LErr1 = 152.5 * phiErr1 / (omega * np.cos(phi1)**2)
        LErr2 = 152.5 * phiErr2 / (omega * np.cos(phi2)**2)    
        L0 = np.average(L1)
        L0Err = np.std(L1)
        perm = L2 / L0
        permErr = np.sqrt((LErr2/L0)**2+((L2*L0Err)/(L0**2))**2)
        #print("Perm:")
        #for i in range(len(perm)):
        #    print(f"{perm[i]} ± {permErr[i]}")
        #print(L1)
        #print(f"{np.average(L1) * 1000} ± {np.std(L1) * 1000}")
        if False:
            ax.errorbar(omega, L1, LErr1, marker=".", linewidth=0, markersize=7, elinewidth=1, capsize=4, label="Ohne Kern")
            ax.errorbar(omega, L2, LErr2, marker=".", linewidth=0, markersize=7, elinewidth=1, capsize=4, label="Mit Kern")
            ax.set_ylabel("Induktivität $L$ in H")
        else:
            ax.errorbar(omega, perm, permErr, marker='.', linewidth=0, markersize=7, elinewidth=1, capsize=4, label="Permeabilität")
            perm0 = np.average(perm)
            print(perm)
            #print(f"{perm0} ± {np.std(perm)}")
            #ax.plot([omega[0], omega[-1]], [perm0, perm0], label="Mittelwert")
            ax.set_ylabel("Permeabilität $\mu$ in H/m")
    ax.set_xlabel("Kreisfrequenz $\omega$ in rad/s")
    plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, top=True, bottom=True, left=True, right=True)
    plt.grid()
    plt.legend()
    plt.show()

#teil1()

#print(np.array([2,4,6,8,10]) / np.array([2, 2, 2, 2, 2]))

#phasenverschiebung("2a", 600)
