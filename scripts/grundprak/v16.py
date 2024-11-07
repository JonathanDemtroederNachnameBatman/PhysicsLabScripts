import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import xlwings as xw

mid = 80 + 5/6

def deg(val):
    parts = val.split('°')
    dec = parts[1][0:-1] if parts[1][-1] == "'" else parts[1]
    dec = float(dec)
    dec = dec / 60
    dec = int(parts[0]) + dec
    return round(np.abs(mid - dec), 3)

def tolatex(l):
    res = ''
    for row in l:
        for i in range(len(row)):
            cell = row[i]
            # formattiere wellenlänge wenn es der letzte wert der reihe ist
            if cell is None:
                s = '-'
            elif i == len(row) - 1:
                s = cell
            else:
                s = str(cell)
            res += s + ' & '
        res = res[0:-2] + '\\\\ \\hline\n'
    return res

def calc_λ(r):
    # berechnet wellenlängen und bildet den mittelwerrt
    data = []
    error = []
    for i in range(1, len(r) - 1):
        if r[i] is not None:
            data.append(np.sin(np.deg2rad(r[i])) * 2e-6 / np.ceil(i / 2))
            error.append((5.0/60.0) * np.cos(np.deg2rad(r[i])) * 2e-6 / np.ceil(i / 2))
            
    return [np.mean(data), np.mean(error)]

def parse_data(data):
    pd = []
    for row in data:
        first = row[0]
        if isinstance(first, str) and not first[0].isnumeric():
            color = first[0:-1]
            prow = np.repeat(None, 8)
            prow[0] = color
            pd.append(prow)
        else:
            order = -1
            right = True
            for cell in row:
                if isinstance(cell, float):
                    order = int(cell)
                elif isinstance(cell, str):
                    if "°" in cell:
                        if order > 0:
                            index = (order - 1) * 2 + 1
                            if right:
                                right = False
                                index += 1
                            if not pd[-1][index]:
                                pd[-1][index] = deg(cell)
                    else:
                        order = int(cell[0])
    return np.array(pd)


data = xw.Book('data/Messwerte_V16.xlsx').sheets[0]

# Teil 1
pd = parse_data(data.range('C7:F37').value)
λs = []

for r in pd:
    λs.append(calc_λ(r))
    r[-1] = f'{np.round(1e9 * λs[-1][0], 1)} \pm {np.round(1e9 * λs[-1][0], 1)}'
#print(pd)
#print(tolatex(pd))

def do_teil_2(data2):
    for j in range(len(data2)):
        r = data2[j]
        data = []
        error = []
        λ = λs[j]
        for i in range(1, len(r) - 1):
            if r[i] is not None:
                m = np.ceil(i / 2)
                data.append(m * λ[0] / np.sin(np.deg2rad(r[i])))
                error.append(np.sqrt(((5.0/60.0)*m*λ[0]*np.cos(r[i])/np.sin(r[i])**2)**2 + (λ[1]*m/np.sin(r[i]))**2))
    return [np.mean(data), np.mean(error)]

pd = parse_data(data.range('C43:F73').value)
#print(pd)
# g wie gitterkonstante
g = do_teil_2(pd)
print(f'{np.round(g[0] * 1e6, 3)}({np.round(g[1] * 1e6, 3)})')
