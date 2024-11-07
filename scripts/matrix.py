import matplotlib.pyplot as plt
import numpy as np
import time
from fractions import Fraction as frac

# 26

zero = frac(0, 1)
one = frac(1, 1)

def as_frac_matrix(matrix):
    return np.array([([frac(x) for x in row]) for row in matrix])

def clear_column(mat, inverse, n: int):
    rows, columns = mat.shape
    # falls schlüssel wert 0 ist Zeilen tauschen
    if mat[n][n] == zero:
        for i in range(0, columns):
            if mat[i][n] != zero:
                a = mat[n]
                mat[n] = mat[i]
                mat[i] = a
        if mat[n][n] == zero:
            # keine geignete Zeile gefunden
            print("Computer sagt nein")
            return

    # wenn Schlüsselwert nicht 1 ist, die ganze Zeile so multiplizieren, dass er 1 wird
    keyVal = mat[n][n]
    if keyVal != one:
        mod = keyVal ** -1
        if keyVal > 0:
            mod = -mod
        for i in range(0, columns):
            mat[n][i] = mat[n][i] * mod
            inverse[n][i] = inverse[n][i] * mod

    # jede Zeile so multiplizieren das der Wert in Spalte n 0 wird
    for row in range(0, rows):
        if row != n:
            val = mat[row][n]
            # val + a * key_val = 0
            # a = - val/key_val  // key_val ist der diagonal Wert und ist hier immer 1 (siehe oben)
            a = -val
            # gesamte reihe der matrix und der resultierenden inversen Matrix multiplizieren
            for column in range(0, columns):
                mat[row][column] = mat[row][column] + a * mat[n][column]
                inverse[row][column] = inverse[row][column] + a * inverse[n][column]

def invert(mat):
    rows, columns = mat.shape
    inverse = as_frac_matrix(np.identity(rows))

    if rows == 0 or columns == 0 or rows != columns:
        print("Idiot")
        return

    for i in range(0, columns):
        clear_column(mat, inverse, i)
        #print("")
        #print("Step {}", i + 1)
        #print(mat)

    return inverse    

# Aufgaben Teil i
matrix = [
    [11*np.e, 0, 3*np.e],
    [0, 3, 0],
    [2, 0, 6]
]
matrix = as_frac_matrix(matrix)


print(matrix)
print(invert(matrix))

# Aufgaben Teil ii

def gen_matrix(size: int):
    a = []
    for i in range(0, size):
        b = []
        a.append(b)
        for j in range(0, size):
            b.append(i + 1 if j == i else 1)
    return as_frac_matrix(a)

#x = []
#y = []

#for i in range(1, 60):
    mat = gen_matrix(i)
    now = time.process_time()
    invert(mat)
    end = time.process_time()
    print(end - now)
    x.append(i)
    y.append(end - now)

#plt.plot(x, y, '.')
#plt.title("Gauß Jordan Verfahren")
#plt.xlabel("Matrix Dimension")
#plt.ylabel("Zeit in ms")
#plt.show()*/