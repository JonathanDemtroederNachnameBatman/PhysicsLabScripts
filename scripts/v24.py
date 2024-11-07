import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import optimize
from lib import helpers as hp

L1 = 15e-3
L2 = 12e-3
g = 63e-3
b = 417e-3
egb = 2e-3
dO = 20.5e-3
edO = 0.5e-3
eL = 1e-3
f = 5.0/8.0
ef = 1.0/8.0
AL = f * (L1*L2*g**2) / (b**2)
#eAL = np.sqrt((eL * L2*g**2 / (2*b**2))**2 + (eL * L1*g**2 / (2*b**2))**2 + (egb * L1*L2*g / (b**2))**2 + (egb * L1*L2*g**2 / (b))**2)
eAL = np.sqrt((eL * f*L2*g**2 / (b**2))**2 + (eL * f*L1*g**2 / (b**2))**2 + (egb * 2*f*L1*L2*g / (b**2))**2 + (egb * 2*f*L1*L2*g**2 / (b))**2 + (ef * L1*L2*g**2 / (b**2))**2)
print(f'AL: {round(AL * 1e6,3)}\pm{round(eAL*1e6,3)}')

AO = np.pi * dO**2 / 4
eAO = np.pi * dO * edO / 2

print(f'AO: {round(AO * 1e6,3)}\pm{round(eAO*1e6,3)}')

IO = 3.07e-3
IL = 7.16e-3
eI = 0.01e-3
rO = 38e-3
rL = 375e-3
er = 1e-3
K  = -273.15
TO = 509+K
eTO = 1

TL4 = TO**4 * (IL * AO * rL**2) / (IO * AL * rO**2)
eTL4 = np.sqrt((eI * TL4/IL)**2 + (eAO*TL4/AO)**2 + (er*2*TL4/rL)**2 + (eTO+4*TL4/TO)**2 + (eI*TL4/IO)**2 + (eAL*TL4/AL)**2 + (er*2*TL4)**2)
TL = TL4**(0.25)
eTL = eTL4 / (4*TL**3)
print(f'TL: {round(TL,3)}\pm{round(eTL,3)}')

def l(beta):
    return 1e-9 * (beta + 3.921) / 7.788e-3

w1 = 2.5
w2 = 2.6
ew = 0.05
a = 7.788e-3
b = -3.921
eb = 0.018
print(f'l1 {l(w1)} | l2 {l(w2)}')
T1 = TL
T2 = T1 * (w1-b)/(w2-b)
eT2 = np.sqrt((eTL*(w1-b)/(w2-b))**2 + (TL*ew/(w2-b))**2 + (TL*ew*(w1-b)/(w2-b)**2)**2 + (TL*eb*(w2-w1)/(w2-b)**2))
print(f'T2: {round(T2,3)}\pm{round(eT2,3)}')
print(f'Wien T1 {2900e-6 / l(w1)}')
print(f'Wien T2 {2900e-6 / l(w2)}')
l = 1e9 * 2900e-6 / TL
print(f'Wien l1 {l}')
print(l * a+b)
