import numpy as np
from numpy.linalg import eigh,norm
import random
from numpy.random import choice
import pylab as pl
from bitstring import BitArray,BitStream


np.set_printoptions(precision=2,threshold=5,edgeitems=10,linewidth=180)

vons = [np.array([1,0]),np.array([0,1])]                         # a 2D Hilbert space basis
v4 = [np.kron(vons[i],vons[j]) for i in range(2) for j in range(2)]   # a corresponding 4D tensor product basis vi x vj of H x H
for i in v4: print(i)


A = np.array([[17,4],[4,17]]); print(A,'\n')
E = np.eye(2,2,dtype=int)
EA = np.kron(E,A); print(EA,'\n')
AE = np.kron(A,E); print(AE)


# (1 x A)(v1 x v1)
EA.dot(v4[0])
# (A x 1)(v1 x v1)
AE.dot(v4[0])


for i in range(16):
    psi = np.zeros(16)
    psi[i] = 1
    print(i,'\t',np.base_repr(i).zfill(4),'\t',psi)
print()


# All counter states with calc bit down (num, bit-reps, Hilbert spc vector)
for i in range(0,16,2):
    psi = np.zeros(16)
    psi[i] = 1
    print(i,'\t',np.base_repr(i).zfill(4),'\t',psi)
print()


# All counter states with calc bit down (num, bit-reps, Hilbert spc vector)
for i in range(0,16,2):
    psi = np.zeros(16)
    psi[i+1] = 1
    print(i+1,'\t',np.base_repr(i+1).zfill(4),'\t',psi)
print()


# The individual gate & counter ops
g = np.array([[1+1j,1-1j],[1-1j,1+1j]])/2 # sqrtnot gate
c = np.array([[0,0],[1,0]])               # c creator (1,0)='runter' -> (0,1)='rauf' 
d = c.T                                   # d destructor (0,1)->(1,0)
e = np.eye(2,2)                           # id

# Properties if sqrt(NOT)
print(g.dot(g.conj().T))
print(g.dot(g))
print()


# The tensor products of the operators into (3+1)*2-dim space
d0=np.kron(d,np.kron(e,np.kron(e,e))) # site 0, idx 0,1 a_0
d1=np.kron(e,np.kron(d,np.kron(e,e))) # site 1, idx 2,3 a_1
c1=np.kron(e,np.kron(c,np.kron(e,e))) # site 1, idx 2,3 a+_1
c2=np.kron(e,np.kron(e,np.kron(c,e))) # site 2, idx 4,5 a+_2
sn=np.kron(e,np.kron(e,np.kron(e,g))) # 6,7 sqrtnot gate


def StateToBin(psi):
    return np.base_repr(int(np.sum(psi*np.array([i for i in range(16)])))).zfill(4)

# Let's act on the states with creation / destruction ops
psi = np.zeros(16)
psi[0b0000] = 1                       # all counter sites = 'down', calc bit = 'down'. state *NEVER* occurs in calc.
print(psi,'\t',StateToBin(psi))
psu = (d0.T).dot(psi)                 # flip 0th counter qubit up
print(psu,'\t',StateToBin(psu))
psa = c1.dot(d0.dot(psu))             # shift an up counter at site 0 by one site from site 0->1 
print(psa,'\t',StateToBin(psa))
# ----------------
pso = c1.dot(d0.dot(sn.dot(psu)))     # shift an up counter by one site from site 0->1 *AND* apply SNOT gate to calc bit
print(pso)
# Note that the state pso can only be a superposition of the states |u1> = '0100' and |u2> = '0101'
# Check this
u1 = np.zeros(16)
u1[0b0100] = 1
u2 = np.zeros(16)
u2[0b0101] = 1
tessi = u1.dot(pso)*u1 + u2.dot(pso)*u2   # basis reps. of pso. Note: u_i normalized
print(max(abs(tessi - pso)))


# Zeit entwicklung

# Set up the Hamiltonian
a =c2.dot(d1).dot(sn) # (s+_2 * s-_1 * sn)
a+=np.conj(a.T)       # + h.c.
h =c1.dot(d0).dot(sn) # (s+_1 * s-_0 * sn)
h+=np.conj(h.T)       # h.c.
h+=a                  # Hamiltonian

# Diagonalize H: E=vd.H.v
e,v = eigh(h)
vd = np.conj(v.T)

def Run(psi,t):
    return v.dot(np.diag(np.exp(-1j*e*t)).dot(vd.dot(psi)))


# clarify action of random.choices
distri = {'beer':0,'wine':0,'whiskey':0}

for i in range(1000):
    distri[random.choices(['beer','wine','whiskey'],weights=[.6,.3,.1])[0]] += 1

print(distri)


def Measure(psi):
    ''' Eval the probabilities P[0..7] of the counter values. We know that many counter
        values will never arise. But we eval them anyway for display purposes. Symmetry
        classification of H would totally go against 'explaining' things ;)  
        Once we have P we draw the intermediate step according to P. Then we project onto
        the corresponding subspace and normalize. The cat is definitely dead / alive ;)
        If P is in 3rd position stop, if not continue.
    '''
    P = np.zeros(8)
    for c in range(0,16,2):                 # loop 8 counter subspaces
        for r in range(2):                  # loop r-qbit states
            u = np.zeros(16)                # qc-basis state
            u[c+r] = 1                      # c+r is idx of basis state
            # - peek --
            #print((c,r,u))
            P[c//2] += abs(u.dot(psi))**2   # ok. QM I ;) P contains 8 propabilities
    P[abs(P)<1e-14] = 0                     # garbage = 0 
    # - peek --
    #print(P)
    # Now do the MAGIC QUANTUM MECHANICAL MEASUREMENT ;)
    # 1st: You only measure counter state with propabilty (random.choices is fastest!)
    cs = random.choices([0,2,4,6,8,10,12,14],weights=P)[0] # the counter state
    # - peek --
    #print(cs)
    # 2nd: Once you got a defined counter state - project QC onto that state.
    # In doing so project onto both calc bit basis states
    u = np.zeros(16)
    u[cs] = 1
    psim = u * (u.dot(psi))
    u[cs] = 0
    u[cs+1] = 1
    psim += u * (u.dot(psi))
    psim /= norm(psim)
    psim[abs(psim)<1e-16] = 0               # num garbage = 0 
    return psim,cs #,P


def DrawQCStateProp(phi,upd=True):
    ''' (Re)Draw propability amplitudes of complete QC state. This function is very badly coded.
        It assumes existence of two global namespace variables, i.e. fig1, ax1. The reason
        for doing this is to crash the code if the figure has not been generated previously.
    '''
    if upd:
        ax1.clear() # clear plot
    ax1.bar(np.arange(0,16),abs(phi)**2,width=.2)
    ax1.set_xlim([-1,16])
    ax1.set_ylim([0,1.1])
    #fig1.show()
    fig1.canvas.draw()



fig1=pl.figure(figsize=(7,2))
ax1=fig1.add_subplot(111)


# Do a run of our QC
psi = np.zeros(16)
psi[0b1001] = 1  # 1st counter site =1, calc bit =1
#DrawQCStateProp(psi)


psi = np.zeros(16)
psi[0b1001] = 1
#DrawQCStateProp(psi)


phi = Run(psi, 1.2)
#DrawQCStateProp(phi)

psi, c = Measure(phi)
print(c)
DrawQCStateProp(psi)
if c == 2:
    print(abs(psi))

pl.show()
