import numpy as np

def clear_column(p, a, l, u, i, n):
    print()
    print(f"Clear col {i}")
    key_val = u[i][i]

    if key_val == 0:
        found = False
        biggest = -1
        index = -1
        for j in range(i + 1, n):
            val = u[j][i]
            if val != 0 and (not found or val > biggest):
                biggest = val
                index = j
                found = True
        if not found:
            raise RuntimeError("Could not find proper row vector to switch!")
        p[[i, index]] = p[[index, i]]
        u[[i, index]] = u[[index, i]]
        key_val = u[i][i]


    print(f'Key val {key_val}')
    for row in range(i + 1, n):
        if row != i:
            val = u[row][i]
            # val + a * key_val = 0
            a = -val / key_val
            for column in range(0, n):
                u[row][column] = u[row][column] + a * u[i][column]
                l[row][column] = l[row][column] - a * l[i][column]
    print(l)
    print(u)


a = np.array([
    [1, 1, 1],
    [1, 2, 1],
    [2, 7, 9]
])
rows, cols = a.shape
if rows != cols:
    raise RuntimeError('Matrix muss quadratisch sein')
n = rows
p = np.identity(n)
l = p.copy()
u = a.copy()

for i in range(0, n):
    clear_column(p, a, l, u, i, n)

print(p)
print(a)
print(l)
print(u)

print(p.dot(a))
print(l.dot(u)) # irgendwas funzt nicht richtig
