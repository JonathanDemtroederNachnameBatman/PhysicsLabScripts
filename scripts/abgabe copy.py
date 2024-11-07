import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, Locator)
    
particles = 5000 #n_p
dx = 2.
nx = 128 
vth = 0.1
#ub = 4
dt = 0.5
nt = 256

tt = dt * np.arange(nt)

box = 'Box-Car'
tri = 'Triangular'
modes = [box, tri]

fft = 'FFT'
eul = 'Euler'
l_f = 'Leap-Frog'
e_field_modes = [fft, eul, l_f]

def build_data(instability, e_field_method: str, density_method: str, velocity_method: str):
    if not density_method in modes: raise Exception('Only box and triangle allowed')
    if not velocity_method in modes: raise Exception('Only box and triangle allowed')
    if not e_field_method in e_field_modes: raise Exception(f'Only {e_field_modes} allowed')
    print(f'Building data with instability {instability}, e field method {e_field_method}, density method {density_method}, velocity method {velocity_method}')
    
    system_size = float(dx * nx) # lx
    x_grid = dx * np.arange(nx)
    xt_particles = np.linspace(0, system_size - system_size / particles, particles)
    # particle velocity at current t
    vt_particles = np.random.normal(loc=0., scale=vth, size=particles) + np.power(-1, np.arange(particles)) * instability
    # frequency from fourier transform
    kk = (2.*np.pi) * np.fft.fftfreq(nx, d=dx)
    # can't divide by 0, so we make it small
    kk[0] = 1.e-6

    # time dependent plot data
    x_particles = np.zeros((nt, particles))
    density_particles = np.zeros((nt, nx))
    e_field = np.zeros((nt, nx))
    v_particles = np.zeros((nt, particles))

    # compute time evolution of position, density, e-field and velocity
    r = range(nt)
    #r = range(100, 101) # for testing
    for it in r:
        print(f' - time evolution of frame {it}/{nt}   ', end='\r')
        # move particles with current velocity
        xt_particles = xt_particles + dt*vt_particles
        # wrap around particles
        xt_particles[xt_particles > system_size] = xt_particles[xt_particles > system_size] - system_size
        xt_particles[xt_particles < 0] = xt_particles[xt_particles < 0] + system_size
        # save particle position to time frame
        x_particles[it] = np.copy(xt_particles)

        ## density
        density = np.zeros(nx)
        if density_method == box:
            # evaluate density with box function
            for ip in range(particles):
                ix = np.mod(int(np.round(xt_particles[ip] / dx)), nx)
                density[ix] += 1
        elif density_method == tri:
            # evaluate density with triangle function
            for ip in range(particles):
                ixm = int(np.floor(xt_particles[ip]/dx))
                ixp = np.mod(ixm+1,nx)
                wxp = xt_particles[ip]/dx-ixm; wxm=1.-wxp
                density[ixm] += wxm; density[ixp] += wxp
                
        # normalize density
        density = density * nx / particles
        # save particle density to time frame
        density_particles[it] = density
        
        ## E - Field
        if e_field_method == fft: 
            # calculate e field via fourier trafo of the particle density
            ex = 1j * np.fft.fft(density) / kk
            ex[0] = 0. 
            ex[int(nx/2)] = 0.
            ex = np.real(np.fft.ifft(ex))
        elif e_field_method == eul:
            # calculate the e field via solving the differential equation with euler method
            ex = np.zeros(nx)
            ex[0] = 0.
            for ix in range(nx-1):
                ex[ix+1] = ex[ix] + dx*(1. - density[ix])
            ex = ex - np.sum(ex)/nx
        elif e_field_method == l_f:
            # calculate the e field via solving the differential equation with leap-frog method
            ex = np.zeros(nx)
            ex[0] = 0.
            ex[1] = ex[0] + dx*(1. - density[0])
            for ix in range(1, nx-1):
                ex[ix+1] = ex[ix-1] + 2*dx*(1. - density[ix])
            ex = ex - np.sum(ex)/nx
        # save e field to time frame
        e_field[it] = ex
        
        # Velocity
        if velocity_method == 'box':
            for ip in range(particles):
                ix=np.mod(int(np.round(xt_particles[ip] / dx)), nx)
                vt_particles[ip] = vt_particles[ip] - dt * ex[ix]
        else:
            for ip in range(particles):
                ixm=int(np.floor(xt_particles[ip]/dx))
                ixp=np.mod(ixm+1,nx)
                wxp=xt_particles[ip] / dx-ixm
                wxm=1.-wxp
                vt_particles[ip] = vt_particles[ip] - dt * (wxm * ex[ixm] + wxp * ex[ixp])
        v_particles[it] = np.copy(vt_particles)

    return (x_particles, density_particles, e_field, v_particles)
    
def calc_energy(instability, e_field_method: str, density_method: str, velocity_method: str):
    x_particles, density_particles, e_field, v_particles = build_data(instability, e_field_method, density_method, velocity_method)
    return np.array([np.sum(v_particles[it] ** 2) / particles + np.sum(e_field[it] ** 2) / nx for it in range(len(v_particles))])

def make_plot(fig, i, instability, e_field_method: str, density_method: str, velocity_method: str):
    #if i > 1: return None
    tick_labels = i == 4 or i == 8
    E = calc_energy(instability, e_field_method, density_method, velocity_method)
    ax = fig.add_subplot()
    ax.set_title(f'E Field: {e_field_method}, Dens.: {density_method[:3]}, Vel.: {velocity_method[:3]}', fontsize=7.5)
    # Dekoration & Positionierung
    # ...
    ax.plot(tt, E)
    
    return ax

def plot_run(instability):
    cm = 1/2.54
    fig = plt.figure(figsize=(15*cm, 15*cm))
    fig.supylabel('Gesamtenergie $E$')
    fig.supxlabel('$t$')
    dif = l_f
    ax = make_plot(fig, 1, instability, dif, box, box)
    ax = make_plot(fig, 2, instability, dif, tri, box)
    ax = make_plot(fig, 3, instability, dif, box, tri)
    ax = make_plot(fig, 4, instability, dif, tri, tri)
    ax = make_plot(fig, 5, instability, fft, box, box)
    ax = make_plot(fig, 6, instability, fft, tri, box)
    ax = make_plot(fig, 7, instability, fft, box, tri)
    ax = make_plot(fig, 8, instability, fft, tri, tri)
    
    file = f'inst_{instability}' if instability != 0 else 'thermal'
    plt.savefig(f'energy_{file}')
    plt.show()
    
#plot_run(0)
plot_run(4)
