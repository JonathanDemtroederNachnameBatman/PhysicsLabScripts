import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from lib import helpers as hp

n = np.array([1.000270, 1.000544, 1.000056, 1.000081])
ne = np.array([0.000010, 0.000012, 0.000013, 0.000027])
p = np.array([1,2,1,2])
ep = 0.0049
T = 293
eT = 5
kb = 1.38064852e-3 # e-20
eps0 = 8.85418781e-2 # e-10
f = 1e5

a = (n**2-1)/(n**2+2) * 3*eps0*kb*T/p * f # in e-40
ea = np.sqrt( ((6*n)/(n**2+2)**2 * 3*eps0*kb*T/p * f * ne)**2 + \
             ((n**2-1)/(n**2+2) * 3*eps0*kb/p * f * eT)**2 + \
             ((n**2-1)/(n**2+2) * 3*eps0*kb*T/p**2 * f * ep)**2 )

print(a)
print(ea)

