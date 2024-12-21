import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import scipy.signal
from scipy import optimize
from scipy import signal
from scipy import stats

from etc import helpers as hp

def read(file):
    return pd.read_csv(f'../../data/fprak_3/{file}.txt', decimal=',', header=None, dtype=np.float64, skiprows=6, sep=r'\s+').to_numpy().T

def read_fld(file):
    return pd.read_csv(f'../../data/fprak_3/{file}.fld', decimal='.', header=None, dtype=np.float64, skiprows=0, sep=r'\s+').to_numpy().T

def read_fit(file):
    return pd.read_csv(f'../../data/fprak_3/{file}_fit.txt', decimal='.', header=None, dtype=np.float64, skiprows=1, sep=r'\s+').to_numpy().T

def save(file):
    plt.savefig(f'../../tmp/fprak_3_{file}.svg', bbox_inches='tight')

def save_arr(a, file):
    f = f'../../data/fprak_3/{file}.fld'
    np.savetxt(f, a.T, fmt=['%f', '%.0f'], delimiter='\t')
    #with open(f, 'r') as file:
    #    s = file.read()
    #s = s.replace('.', ',')
    #with open(f, 'w') as file:
    #    file.write(s)


fp = 512 + 1

def fold(D, folding_point=fp):
    counts1 = D[1][:folding_point]
    counts2 = D[1][folding_point:][:folding_point]
    if len(counts1) > len(counts2):
        counts1 = counts1[len(counts1)-len(counts2):]
    return np.cos(D[0][:len(counts1)]/(fp-1) * np.pi), counts1 + np.flip(counts2)

def to_fdl(name, v):
    D = read(name)
    print(D)
    rel_velocity, counts = fold(D)
    velocity = rel_velocity * (v*0.16283)
    print(velocity)
    save_arr(np.array([velocity, counts]), name)

def find_fp():
    n = 'eisenfolie'
    D = read(n)
    i_min = -1
    c_min = -1
    for i in np.arange(507, 518):
        rel_velocity, counts = fold(D, folding_point=i)
        #fig, ax = plt.subplots()
        #ax.config(hp.AxisConfig(), hp.AxisConfig())
        #ax.plot(rel_velocity, counts, label=f'{n} - {i-1}')
        #ax.legend()
        #save(f'fe_2O_3_{i-1}')
        c = np.min(counts)
        if c_min < 0 or c < c_min:
            c_min = c
            i_min = i

    print(f'Best fp at {i_min-1} and {c_min}')
    rel_velocity, counts = fold(D, folding_point=i_min)
    #fig, ax = plt.subplots()
    #ax.config(hp.AxisConfig(), hp.AxisConfig())
    #ax.plot(rel_velocity, counts)
    plt.show()

def convert():
    to_fdl('eisenfolie', 90.4)
    to_fdl('fe_2O_3', 80.0)
    to_fdl('fe_3O_4', 80.0)
    to_fdl('gelbes_bls', 20.0)
    to_fdl('rotes_bls', 8.2)
    to_fdl('stahl', 20.2)

#convert()

def fe3o4():
    # B1 = 46.0084
    # B2 = 28.7608
    # Vzz1 = -0.119969
    # Vzz2 = -157.043
    # eta1 = 310.273
    # eta2 = 5.52751
    # theta1 = -89.9475
    # theta2 = 53.3997
    # phi1 = 136.668
    # phi2 = -135.514
    # CS1 = 0.659553
    # CS2 = 0.306643
    # omega1 = 0.184397
    # omega2 = 0.16029
    # I0 = 148417
    # A1 = 35402.3
    # A2 = 19936.9

    D = read_fit('fe_3O_4')
    fig, ax = plt.subplots(figsize=(16/2.54, 10/2.54))
    x = D[0]
    y = D[1]

    ax.config(hp.AxisConfig(majors=3, minors=6, label='v in $\\frac{mm}{s}$', min=D[0][0], max=D[0][-1]), hp.AxisConfig(label='Intensität', min=124000, max=150000))
    ax.plot(x, y, '.k', markersize=2, linewidth=1, label='Messdaten')
    ax.plot(D[4], D[5], markersize=2, linewidth=1, label='Subspektrum 1')
    ax.plot(D[6], D[7], markersize=2, linewidth=1, label='Subspektrum 2')
    ax.plot(D[2], D[3], markersize=2, linewidth=1, label='Gesamtspektrum')
    plt.legend()
    save('fe3o4')
    #plt.show()

# B =
# Vzz =
# eta =
# theta =
# phi =
# CS =
# omega =
# I0 =
# A =

def fe2o3():
    D = read_fit('fe_2O_3')
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    x = D[0]
    y = D[1]

    ax.config(hp.AxisConfig(majors=3, minors=6, label='v in $\\frac{mm}{s}$', min=D[0][0], max=D[0][-1]),
              hp.AxisConfig(label='Intensität', min=30000, max=38500))
    ax.plot(x, y, '.k', markersize=2, linewidth=1, label='Messdaten')
    ax.plot(D[2], D[3], markersize=2, linewidth=1, label='Fit')
    plt.legend()
    save('fe2o3')
    plt.show()

def rotes_bls():
    D = read_fit('rotes_bls')
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    x = D[0]
    y = D[1]

    ax.config(hp.AxisConfig(majors=0.2, minors=2, label='v in $\\frac{mm}{s}$', min=D[0][0], max=D[0][-1]),
              hp.AxisConfig(label='Intensität', min=34000, max=39700))
    ax.plot(x, y, '.k', markersize=2, linewidth=1, label='Messdaten')
    ax.plot(D[2], D[3], markersize=2, linewidth=1, label='Fit')
    plt.legend()
    save('rotes_bls')
    plt.show()

def gelbes_bls():
    D = read_fit('gelbes_bls')
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    x = D[0]
    y = D[1]

    ax.config(hp.AxisConfig(majors=1, minors=4, label='v in $\\frac{mm}{s}$', min=D[0][0], max=D[0][-1]),
              hp.AxisConfig(label='Intensität', min=24000, max=31000))
    ax.plot(x, y, '.k', markersize=2, linewidth=1, label='Messdaten')
    ax.plot(D[2], D[3], markersize=2, linewidth=1, label='Fit')
    plt.legend()
    save('gelbes_bls')
    plt.show()

def stahl():
    D = read_fit('stahl')
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    x = D[0]
    y = D[1]

    ax.config(hp.AxisConfig(majors=1, minors=4, label='v in $\\frac{mm}{s}$', min=D[0][0], max=D[0][-1]),
              hp.AxisConfig(label='Intensität', min=2500, max=4500))
    ax.plot(x, y, '.k', markersize=2, linewidth=1, label='Messdaten')
    ax.plot(D[2], D[3], markersize=2, linewidth=1, label='Fit')
    plt.legend()
    save('stahl')
    plt.show()

def hex_lorentz(x, A, I, omega, a1, a2, a3, a4, a5, a6):
    return I + A * omega**2 * (3 / ((x-a1)**2+omega**2) + 2 / ((x-a2)**2+omega**2) + 1 / ((x-a3)**2+omega**2) + 1 / ((x-a4)**2+omega**2) + 2 / ((x-a5)**2+omega**2) + 3 / ((x-a6)**2+omega**2))

def eisenfolie():
    D = read('eisenfolie')
    rel_velocity, counts = fold(D)
    rel_velocity = -rel_velocity
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))

    #popt, pcov = optimize.curve_fit(hex_lorentz, rel_velocity, counts, p0=[-200, 67000, 0.05, -0.38, -0.21, -0.05, 0.04, 0.20, 0.38], bounds=([-10000, 66000, 0, -0.4, -0.3, -0.1, 0.0, 0.1, 0.3], [0, 68000, 1, -0.3, -0.1, 0.0, 0.1, 0.3, 0.4]))
    #popt = [-500, 67000, 0.05, -0.38, -0.21, -0.05, 0.04, 0.20, 0.38]
    #print(popt)
    #print(np.sqrt(np.diag(pcov)))
    popt = np.array([-4.99812291e+03, 6.71493343e+04, 1.16919476e-02, -3.61188848e-01, -2.10607699e-01, -5.92198493e-02, 5.38229813e-02, 2.06000979e-01, 3.57844582e-01])
    epopt = np.array([5.69404019e+01, 1.88690050e+01, 1.90030854e-04, 2.32294490e-04, 3.55775353e-04, 7.19460913e-04, 7.21359121e-04, 3.57249654e-04, 2.32111853e-04])
    X = np.linspace(rel_velocity[0], rel_velocity[-1], 1000)
    Y = hex_lorentz(X, *popt)

    text = hp.param_text_fit(['a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6'], popt[3:], 4)

    ax.config(hp.AxisConfig(majors=0.2, minors=2, label='relative Geschwindigkeit', min=rel_velocity[0], max=rel_velocity[-1]),
              hp.AxisConfig(label='Intensität'))
    ax.plot(rel_velocity, counts, '.k', markersize=2, linewidth=1, label='Messdaten')
    ax.plot(X, Y, label='Fit', linewidth=1)
    ax.text_box('Peak Positionen:\n' + text, 0.975, 0.05, vertical='bottom', horizontal='right')
    plt.legend()
    save('eisenfolie')
    plt.show()

    upper_pos = popt[3:][3:]
    lower_pos = np.flip(popt[3:][:3])
    dvr = upper_pos - lower_pos
    dv = [1.677, 6.167, 10.6570]
    l = stats.linregress(dvr, dv)
    print(l.slope, l.stderr, l.rvalue)
    fig, ax = plt.subplots()
    ax.config(hp.AxisConfig(), hp.AxisConfig())
    ax.plot(dvr, dv, '.k')
    X = np.array([dvr[0], dvr[-1]])
    ax.plot(X, X*l.slope + l.intercept, '--')
    plt.show()

    c = 14.819
    lower_pos *= c
    upper_pos *= c
    print(lower_pos, upper_pos, dvr)
    cs = (lower_pos + upper_pos) / 2
    print(cs)
    print(np.average(cs), np.std(cs))

def calc_effects(name, CS, eCs, B, eB, Vzz, eVzz, eta, eeta):
    q = 0.0167*Vzz*np.sqrt(1+(eta**2)/3)
    eq = np.sqrt((0.0167*eVzz*np.sqrt(1+(eta**2)/3))**2 + (0.0167*Vzz*eta*eeta/(3*np.sqrt(1+(eta**2)/3)))**2)
    z12 = 0.65572 * 0.18088 * B
    ez12 = 0.65572 * 0.18088 * eB
    z32 = 0.65572 * (-0.10327) * B
    ez32 = 0.65572 * (-0.10327) * eB
    print(name)

    #print(f'Isomerieverschiebung: {CS:.5f} \\pm {eCs:.5f}')
    #print(f'Quadrupolaufspaltung: {q:.5f} \\pm {eq:.5f}')
    #print(f'Zeeman-Aufspaltung 1/2: {z12:.5f} \\pm {ez12:.5f}')
    #print(f'Zeeman-Aufspaltung 3/2: {z32:.5f} \\pm {ez32:.5f}')
    col0 = ['$\\Delta v_i$', '$\\Delta v_q$', '$\\Delta v_{z,1/2}$', '$\\Delta v_{z,3/2}$']
    print(hp.to_latex_table([CS, q, z12, z32], [eCs, eq, ez12, ez32], precision=5, col0=col0))
    return [CS, q, z12, z32], [eCs, eq, ez12, ez32]

def calc_effects_all():

    calc_effects('Rotes Bls', -0.22271, 0.0018, 0.28, 0.78, 16.64, 11.95, 0.3162, 9.0373)
    calc_effects('Gelbes Bsl', -0.11989, 0.00211, -0.32, 0.07, 2.00, 2.02, -0.0005, 1.7842)
    calc_effects('Fe2O3', 0.36445, 0.00193, 51.76, 0.01, -13.06, 0.22, -4, 0.3256)
    calc_effects('Fe3O4 1', 0.659550, 0.001752, 46.01, 0.01, -0.12, 0.01, 310.27290, -1.0)
    calc_effects('Fe3O4 2', 0.306640, 0.002764, 28.76, 0.02, -157.04, 0.10, 5.5275, 0.00405)
    calc_effects('Stahl', -0.17627, 0.00295, -0.37, 0.14, 10.31, 0.74, 0.3580, 0.2787)

#find_fp()
#fe3o4()
#fe2o3()
#rotes_bls()
#gelbes_bls()
#stahl()
eisenfolie()

#calc_effects_all()
