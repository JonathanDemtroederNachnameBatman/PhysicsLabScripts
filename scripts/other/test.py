import matplotlib.pyplot as plt
import numpy as np
import time

'''
x = -1.9
def fact(n):
    r = 1
    for i in range(2, n):
        r = i * r
    return r

def sum(f, n):
    result = 0
    for i in range(1, n):
        result += f(i)
    return result

def piece(k):
    return (16*k+np.sqrt(2)) / (4*(k**2)-1)

arr = [piece(3)]
#j = 10

for k in range(1, 5000):
    arr.append(arr[k-1] + piece(k))

print(arr[48])
plt.plot(arr, '.')
plt.show()  
'''

def gcd2(a, b, rtol=1e-5, atol = 1e-8):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol and abs(b) > 1:
        a, b= b, a % b
    return a

def index(a):
    r = -1
    n = -1
    for i in range(1, int(min(np.abs(a))) + 1):
        b = a[0] / i
        d = np.abs((a[1:] / b) % 1.0 - 0.5)
        c = np.product(d * 2)
        print(i, b, d, c)
        if c > r:
            r = c
            n = i
    i = np.zeros(len(a))
    i[0] = n
    g = a[0] / n
    for j in range(1, len(a)):
        i[j] = int(round(a[j] / g))
    return i

print(np.geomspace(1, 1000))
