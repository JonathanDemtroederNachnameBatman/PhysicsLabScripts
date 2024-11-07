import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

theta_0 = 0.000158
theta_0_fehler = 0.00000245
periodendauer_fehler = 0.005
richtmoment = 0.02393
angles_15 = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360]

def parse_list(lst):
    values = list(lst)
    for i in range(len(values)):
        if isinstance(values[i], str):
            values[i] = float(values[i][:-2])
        values[i] = round(values[i], 1) 
    return values

def avr(lst):
    return sum(lst) / len(lst)

def periode(avr):
    return round(avr/20, 3)

def theta(periode):
    return round((richtmoment * periode * periode / (4 * np.pi * np.pi) - theta_0)* 10000, 3)

def theta_fehler(periode):
    return np.round(np.sqrt(np.power((2*periode*richtmoment)/(4*np.pi*np.pi)*periodendauer_fehler, 2) + 
                   np.power((periode*periode)/(4*np.pi*np.pi)*0.00009, 2) + theta_0_fehler * theta_0_fehler) * 10000, 3)

def trägheitsradius(theta):
    return round(1 / np.sqrt(theta), 3)

def generate(rowNum, rowCount):
    latex = ""
    angles = [0, 15, 30, 45, 60, 75]
    radien = []
    error = []
    for i in range(rowCount):
        phi = angles[i]
        latex += '$\SI{%s}{\degree}$ & ' % (phi)
        sheet = pd.read_excel("Messwerte_Versuch4.xlsx", 'Tabelle1', skiprows = rowNum + i - 1, nrows = 0,  usecols= 'C:E', dtype=np.float64)
        values = parse_list(sheet.columns)
        #print(values)
        periodendauer = periode(avr(values))
        trägheitsmoment = theta(periodendauer)
        trägheits_fehler = theta_fehler(periodendauer)
        radius = trägheitsradius(trägheitsmoment / 10000)
        radius_fehler = np.round((trägheits_fehler/10000) / (2* np.power(trägheitsmoment/10000, 3.0/2.0)), 3)
        radien.append(radius)
        error.append(radius_fehler)
        latex += '$\SI{%s(%s)}{\s}$ & ' % (periodendauer, periodendauer_fehler) 
        latex += '$\SI{%s(%s)e-4}{\kg\m\squared}$ & ' % (trägheitsmoment, trägheits_fehler)
        latex += '$\SI{%s(%s)}{\m}$ \\\\ \\hline\n' % (radius, radius_fehler)
    
    sym_angles= [0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1, 0]
    for i in range(len(sym_angles)):
        if i >= len(radien):
            radien.append(radien[sym_angles[i]])
            error.append(error[sym_angles[i]])

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
    ax1.set_ylim(70,90)
    #ax1.set_rorigin(-0.001)
    ax1.set_yticks([75, 78, 81, 84, 87])
    ax1.set_xticks(np.deg2rad(angles_15[:-1]))
    ax1.plot(np.deg2rad(angles_15), radien, '-b', lw=0.6)
    #ax1.plot(np.deg2rad(angles_15), radien, '.')
    ax1.errorbar(np.deg2rad(angles_15), radien, yerr=error, fmt='.r')
    plt.show()

    #plt.polar(np.deg2rad(angles), radien, '.')
    #plt.thetagrids(angles[:-1])
    #plt.rgrids()
    plt.show()

    return latex

def generate2(rowNum, rowCount):
    latex = ""
    angles = [0, 15, 30, 45, 60, 75, 90]
    radien = []
    error = []
    for i in range(rowCount):
        phi = angles[i]
        latex += '$\SI{%s}{\degree}$ & ' % (phi)
        sheet = pd.read_excel("Messwerte_Versuch4.xlsx", 'Tabelle1', skiprows = rowNum + i - 1, nrows = 0,  usecols= 'C:E', dtype=np.float64)
        values = parse_list(sheet.columns)
        #print(values)
        periodendauer = periode(avr(values))
        trägheitsmoment = theta(periodendauer)
        trägheits_fehler = theta_fehler(periodendauer)
        radius = trägheitsradius(trägheitsmoment / 10000)
        radius_fehler = np.round((trägheits_fehler/10000) / (2* np.power(trägheitsmoment/10000, 3.0/2.0)), 3)
        radien.append(radius)
        error.append(radius_fehler)
        latex += '$\SI{%s(%s)}{\s}$ & ' % (periodendauer, periodendauer_fehler) 
        latex += '$\SI{%s(%s)e-4}{\kg\m\squared}$ & ' % (trägheitsmoment, trägheits_fehler)
        latex += '$\SI{%s(%s)}{\m}$ \\\\ \\hline\n' % (radius, radius_fehler)
    
    sym_angles= [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    for i in range(len(sym_angles)):
        if i >= len(radien):
            radien.append(radien[sym_angles[i]])
            error.append(error[sym_angles[i]])

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
    ax1.set_ylim(70,90)
    #ax1.set_rorigin(-0.001)
    ax1.set_yticks([75, 78, 81, 84, 87])
    ax1.set_xticks(np.deg2rad(angles_15[:-1]))
    ax1.plot(np.deg2rad(angles_15), radien, '-b', lw=0.6)
    ax1.errorbar(np.deg2rad(angles_15), radien, yerr=error, fmt='.r')
    #ax1.plot(np.deg2rad(angles_15), radien, '.')
    plt.show()

    #plt.polar(np.deg2rad(angles), radien, '.')
    #plt.thetagrids(angles[:-1])
    #plt.rgrids()
    plt.show()

    return latex

def generate3(rowNum, rowCount):
    latex = ""
    angles = [0, 15, 30, 45, 60, 75, 90]
    radien = []
    error = []
    for i in range(rowCount):
        #i = rowCount - 1 - i
        phi = angles[i]
        latex += '$\SI{%s}{\degree}$ & ' % (phi)
        sheet = pd.read_excel("Messwerte_Versuch4.xlsx", 'Tabelle1', skiprows = rowNum + i - 1, nrows = 0,  usecols= 'C:E', dtype=np.float64)
        values = parse_list(sheet.columns)
        #print(values)
        periodendauer = periode(avr(values))
        trägheitsmoment = theta(periodendauer)
        trägheits_fehler = theta_fehler(periodendauer)
        radius = trägheitsradius(trägheitsmoment / 10000)
        radius_fehler = np.round((trägheits_fehler/10000) / (2* np.power(trägheitsmoment/10000, 3.0/2.0)), 3)
        radien.append(radius)
        error.append(radius_fehler)
        latex += '$\SI{%s(%s)}{\s}$ & ' % (periodendauer, periodendauer_fehler) 
        latex += '$\SI{%s(%s)e-4}{\kg\m\squared}$ & ' % (trägheitsmoment, trägheits_fehler)
        latex += '$\SI{%s(%s)}{\m}$ \\\\ \\hline\n' % (radius, radius_fehler)
    
    sym_angles= [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    for i in range(len(sym_angles)):
        if i >= len(radien):
            radien.append(radien[sym_angles[i]])
            error.append(error[sym_angles[i]])

    print(error)

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
    ax1.set_ylim(0,70)
    #ax1.set_rorigin(-0.001)
    ax1.set_yticks([10, 20, 30, 40, 50, 60])
    ax1.set_xticks(np.deg2rad(angles_15[:-1]))
    ax1.plot(np.deg2rad(angles_15), radien, '-b', lw=0.6)
    ax1.errorbar(np.deg2rad(angles_15), radien, yerr=error, fmt='.r')
    #ax1.plot(np.deg2rad(angles_15), radien, '.')
    plt.show()

    #plt.polar(np.deg2rad(angles), radien, '.')
    #plt.thetagrids(angles[:-1])
    #plt.rgrids()
    return latex

#print(generate(24, 6))
#print(generate2(30, 7))
print(generate3(41, 7))




