import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, Locator)
import numbers
from scipy.integrate import RK45

# helper class to easily decorate multiple plots with similar data
class AxisConfig:

    def __init__(self, min=None, max=None, minors=None, majors=None, label=None, ticks=True):
        self.min = min
        self.max = max
        self.minors = minors
        self.majors = majors
        self.label = label
        self.ticks = ticks

    def apply(self, axis, set_label):
        if self.majors: 
            if isinstance(self.majors, numbers.Number):
                axis.set_major_locator(MultipleLocator(self.majors))
            elif isinstance(self.majors, Locator):
                axis.set_major_locator(self.majors)
            else: 
                raise Exception('Wrong majors type')
        if self.minors: 
            if isinstance(self.minors, numbers.Number):
                axis.set_minor_locator(MultipleLocator(self.minors))
            elif isinstance(self.minors, Locator):
                axis.set_minor_locator(self.minors)
            else: 
                raise Exception('Wrong minors type')
        if self.label: set_label(self.label, fontsize=10, fontweight='medium')
        if not self.ticks: axis.set_ticklabels([])

# helper method to easily decorate a plot
def config_axis(ax, x: AxisConfig, y: AxisConfig):
    ax.tick_params(direction='in', which='major', left=True, right=True, top=True, bottom=True, length=4, width=1.3)
    ax.tick_params(direction='in', which='minor', left=True, right=True, top=True, bottom=True, length=2, width=1)
    ax.spines[:].set_linewidth(1.3)
    ax.grid()
    x.apply(ax.xaxis, ax.set_xlabel)
    y.apply(ax.yaxis, ax.set_ylabel)
    
# Setup simulation parameters
particles = 100 # particle count
dx = 2. # section with
nx = 128 # section count

dt = 0.5 # discrete time "width"
nt = 256 # discrete time steps or frame amount

tt = dt * np.arange(nt) # time data

# Setup variables for calculation strategies
box = 'Box-Car'
tri = 'Triangular'
modes = [box, tri]

fft = 'FFT'
eul = 'Euler'
l_f = 'Leap-Frog'
#rk4 = 'Runge-Kutta 4'
e_field_modes = [fft, eul, l_f]

# Calculates and returns position, density, e-field and velocity of each particle/section at each frame.
def build_data(instable: bool, e_field_method: str, density_method: str, velocity_method: str):
    """
    Calculates and returns position, density, e-field and velocity of each particle/section at each frame.

    Args:
        instable (bool): thermal or instable run
        e_field_method (str): method used to calculate the e field (FFT, Euler, Leap-Frog)
        density_method (str): method used to calculate density of sections (Box-Car, Triangular)
        velocity_method (str): method used to update velocity of particles (Box-Car, Triangular)

    Raises:
        Exception: if inputs are invalid

    Returns:
        Tuple: a tuple with time-position matrices of position, density, e-field and velocity
    """
    
    # validate input
    if not density_method in modes: raise Exception('Only box and triangle allowed')
    if not velocity_method in modes: raise Exception('Only box and triangle allowed')
    if not e_field_method in e_field_modes: raise Exception(f'Only {e_field_modes} allowed')
    print(f'Building data with instable {instable}, e field method {e_field_method}, density method {density_method}, velocity method {velocity_method}')
    
    system_size = float(dx * nx) # lx
    x_grid = dx * np.arange(nx)
    # particle pos at current t
    xt_particles = np.linspace(0, system_size - system_size / particles, particles)
    # particle velocity at current t
    if instable: vth = 0.1; ub = 4
    else: vth = 1.0; ub = 0
    vt_particles = np.random.normal(loc=0., scale=vth, size=particles) + np.power(-1, np.arange(particles)) * ub
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
    for it in range(nt):
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
                # find index of section for which the shape function is 1
                ix = np.mod(int(np.round(xt_particles[ip] / dx)), nx)
                # increase density for section
                density[ix] += 1
        elif density_method == tri:
            # evaluate density with triangle function
            for ip in range(particles):
                # find indices for which the shape function is > 0
                ixm = int(np.floor(xt_particles[ip]/dx))
                ixp = np.mod(ixm+1,nx)
                # calculate shape function (weight)
                wxp = xt_particles[ip]/dx-ixm; wxm=1.-wxp
                # increase density for section with weight
                density[ixm] += wxm
                density[ixp] += wxp
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
            # inverse fourier
            ex = np.real(np.fft.ifft(ex))
        elif e_field_method == eul:
            # calculate the e field via solving the differential equation with euler method
            ex = np.zeros(nx)
            ex[0] = 0.
            for ix in range(nx-1):
                ex[ix+1] = ex[ix] + dx*(1. - density[ix])
            # normalize
            ex = ex - np.sum(ex)/nx
        elif e_field_method == l_f:
            # calculate the e field via solving the differential equation with leap-frog method
            ex = np.zeros(nx)
            ex[0] = 0.
            # calculate first step with euler method
            ex[1] = ex[0] + dx*(1. - density[0])
            for ix in range(1, nx-1):
                ex[ix+1] = ex[ix-1] + 2*dx*(1. - density[ix])
            # normalize
            ex = ex - np.sum(ex)/nx
        # save e field to time frame
        e_field[it] = ex
        
        # Velocity
        if velocity_method == 'box':
            # calculate velocity for each particle by using the box car shape function
            for ip in range(particles):
                # find index of section for which the shape function is 1
                ix=np.mod(int(np.round(xt_particles[ip] / dx)), nx)
                # increase velocity with e field from section
                vt_particles[ip] = vt_particles[ip] - dt * ex[ix]
        else:
            # calculate velocity for each particle by using the triangular shape function
            for ip in range(particles):
                # find indices for which the shape function is > 0
                ixm=int(np.floor(xt_particles[ip]/dx))
                ixp=np.mod(ixm+1,nx)
                # calculate shape function (weight)
                wxp=xt_particles[ip] / dx-ixm
                wxm=1.-wxp
                # increase velocity with weighted e field from sections
                vt_particles[ip] = vt_particles[ip] - dt * (wxm * ex[ixm] + wxp * ex[ixp])
        # copy and save velocity since we are still using it in next frame
        v_particles[it] = np.copy(vt_particles)

    return (x_particles, density_particles, e_field, v_particles)
    
# calculates energy with or without instability and with given strategies
def calc_energy(instable: bool, e_field_method: str, density_method: str, velocity_method: str):
    # calculate needed data fore each frame first
    x_particles, density_particles, e_field, v_particles = build_data(instable, e_field_method, density_method, velocity_method)
    # calculation of energy made very short with list comprehension
    return np.array([np.sum(v_particles[it] ** 2) / particles + np.sum(e_field[it] ** 2) / nx for it in range(len(v_particles))])

# creates a decorated energy - time plot with given strategies
def make_plot(fig, i, instable: bool, e_field_method: str, density_method: str, velocity_method: str):
    #if i > 1: return None
    tick_labels = i == 4 or i == 8
    E = calc_energy(instable, e_field_method, density_method, velocity_method)
    ax = fig.add_subplot()
    # plot label with strategies
    ax.set_title(f'E Field: {e_field_method}, Dens.: {density_method[:3]}, Vel.: {velocity_method[:3]}', fontsize=7.5)
    config_axis(ax, x=AxisConfig(ticks=tick_labels, majors=50, minors=10), y=AxisConfig(minors=AutoMinorLocator(2)))
    # calculate position in figure
    left = 0.13
    if i > 4:
        left += 0.47 # fft strategy on right side
    p = i % 4
    if p == 0:
        p = 4
    ax.set_position([left, (4-p) * 0.23 + 0.08, 0.38, 0.19])
    ax.plot(tt, E, color='black', linewidth=0.8)
    
    return ax

def plot_run(instable: bool, one_plot: bool=False):
    cm = 1/2.54
    fig = plt.figure(figsize=(15*cm, 15*cm))
    dif = l_f
    fig.supylabel('Gesamtenergie in $E/E_0$')
    fig.supxlabel('Zeit in $t\cdot\omega_P$')
    ax = make_plot(fig, 1, instable, dif, box, box)
    ax = make_plot(fig, 2, instable, dif, tri, box)
    ax = make_plot(fig, 3, instable, dif, box, tri)
    ax = make_plot(fig, 4, instable, dif, tri, tri)
    ax = make_plot(fig, 5, instable, fft, box, box)
    ax = make_plot(fig, 6, instable, fft, tri, box)
    ax = make_plot(fig, 7, instable, fft, box, tri)
    ax = make_plot(fig, 8, instable, fft, tri, tri)
    print('Done                                       ')
    # save figure as png
    file = f'energy_instable' if instable else 'energy_thermal'
    plt.savefig(file)
    #plt.show()

plot_run(False)
plot_run(True)
