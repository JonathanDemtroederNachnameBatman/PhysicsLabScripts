import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

print('Initialisiere')
n_p = 1000
dx = 2.
nx = 128
vth = 1.
ub = 0
dt = 0.5
nt = 256
lx = float(dx * nx)
xx = dx * np.arange(nx)
xp = np.linspace(0,lx-lx/n_p,n_p)
vp = np.random.normal(loc=0., scale=vth, size=n_p) + np.power(-1,np.arange(n_p))*ub
#vp = np.random.normal(loc=0., scale=vth, size=n_p)

kk = (2.*np.pi) * np.fft.fftfreq(nx, d=dx)
kk[0] = 1.e-6

##-- Init figure --##
# x = (minor, major, axis_label, tick_label=False)
def configure_axis(ax, x, y):
    ax.tick_params(direction='in', which='major', left=True, right=True, top=True, bottom=True, length=4, width=1.3)
    ax.tick_params(direction='in', which='minor', left=True, right=True, top=True, bottom=True, length=2, width=1)
    if x[1]: ax.xaxis.set_major_locator(MultipleLocator(x[1]))
    if x[0]: ax.xaxis.set_minor_locator(MultipleLocator(x[0]))
    if y[1]: ax.yaxis.set_major_locator(MultipleLocator(y[1]))
    if y[0]: ax.yaxis.set_minor_locator(MultipleLocator(y[0]))
    if len(x) > 2: ax.set_xlabel(x[2], fontsize=10, fontweight='medium')
    if len(y) > 2: ax.set_ylabel(y[2], fontsize=10, fontweight='medium')
    if len(x) > 3 and not x[3]: ax.xaxis.set_ticklabels([])
    if len(y) > 3 and not y[3]: ax.yaxis.set_ticklabels([])
    ax.spines[:].set_linewidth(1.3)

# Zeitentwickelte plot daten
xp0 = np.zeros((nt, len(xp)))
ds0 = np.zeros((nt, nx))
ex0 = np.zeros((nt, nx))
vp0 = np.zeros((nt, n_p))

def ex_fft(ds):
    exfft = 1j * np.fft.fft(ds) / kk
    exfft[0] = 0. 
    exfft[int(nx/2)] = 0.
    return np.real(np.fft.ifft(exfft))

def ex_diff(ds):
    ex2 = np.zeros(nx)
    ex2[0] = 0.
    for ix in range(nx-1):
        ex2[ix+1] = ex2[ix] + dx*(1. - ds[ix])
    ex2 = ex2 - np.sum(ex2)/nx
    ## periodicity check
    # print(ex2[0], ex2[nx-1]+dx*(1.-ds2[nx-1]))
    return ex2

# berechne Zeitentwicklung vorher um animation wniger zu belasten
r = range(nt)
#r = range(100, 101)
for it in r:
    print(f'Zeitentwicklung von frame {it}', end='\r')
    xp = xp + dt*vp
    xp[xp > lx] = xp[xp > lx] - lx
    xp[xp < 0] = xp[xp < 0] + lx
    xp0[it] = np.copy(xp)
    ## Dichte
    ds = np.zeros(nx)
    for ip in range(n_p):
        ixm=int(np.floor(xp[ip]/dx))
        ixp=np.mod(ixm+1,nx)
        wxp=xp[ip]/dx-ixm; wxm=1.-wxp
        ds[ixm] += wxm; ds[ixp] += wxp
    ds=ds*nx/n_p
    ds0[it] = ds
    ## E - Feld
    #ex = ex_fft(ds)
    ex = ex_diff(ds)
    ex0[it] = ex
    #
    for ip in range(n_p):
        ixm=int(np.floor(xp[ip]/dx))
        ixp=np.mod(ixm+1,nx)
        wxp=xp[ip]/dx-ixm; wxm=1.-wxp
        vp[ip] = vp[ip]-dt*(wxm*ex[ixm]+wxp*ex[ixp])
    vp0[it] = np.copy(vp)

print()
print('Done')

def color_of(i, n):
    h = (4 * (i / float(n))) % 1. # hue
    return matplotlib.colors.hsv_to_rgb([h, 1., 0.75])

def main_plot():
    cm = 1/2.54
    fig = plt.figure(facecolor="white", figsize=(15*cm, 15*cm))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.set_xlim(0, lx)
    ax1.set_ylim(-10, 10)
    ax1.set_position([.1, 0.5, .85, 0.4])
    configure_axis(ax1, (10, 50, '', False), (2.5, 10, r'$v/v_{th}$'))
    ax2.set_xlim(0, lx)
    ax2.set_ylim(0, 2)
    ax2.set_position([.1, .3, .85, .15])
    configure_axis(ax2, (10, 50, '', False), (0.5, 1, r'$n/n_0$'))
    ax3.set_xlim(0, lx)
    ax3.set_ylim(-8, 8)
    ax3.set_position([.1, .1, .85, .15])
    configure_axis(ax3, (10, 50, r'$x/\lambda_D$'), (2.5, 5, r'$E/E_0$'))
    ax1.xaxis

    line1 = ax1.plot([], [], '.', markersize=0.75)[0]
    line2 = ax2.plot([], [], '-')[0]
    line3 = ax3.plot([], [], '-')[0]

    def update(it):    
        title = "$t_i$:" + format(it,"5")
        fig.suptitle(title)
        
        color = color_of(it, nt)

        line1.set_xdata(xp0[it])
        line2.set_xdata(xx)
        line3.set_xdata(xx)
        line1.set_ydata(vp0[it])
        line2.set_ydata(ds0[it])
        line3.set_ydata(ex0[it])
        line1.set_color(color)
        line2.set_color(color)
        line3.set_color(color)
        return [line1, line2, line3],

    # interval = ms zwischen frames
    ani = animation.FuncAnimation(fig=fig, func=update, frames=nt, interval=120, repeat=True)

    def safe(ani):
        print('Start saving')
        ani.save(filename="tmp/cool.gif", writer="pillow")
        print('Done')

    #update(100)
    plt.show()
    #safe(ani)

def xt_plot():
    tt = dt * np.arange(nt)
    cmap = plt.get_cmap('PiYG')
    plt.pcolormesh(xx, tt, ex0, cmap=cmap)
    plt.xlabel('$x/\lambda_D$')
    plt.ylabel('$t\omega_p$')
    cbar = plt.colorbar()
    cbar.set_label('$E/E_0$')
    plt.savefig('tmp/xt_plot.png')
    plt.show()

def k_omega_plot():
    global ex0
    kmin = 2*np.pi/(dx*nx)*(-nx/2)
    kmax = 2*np.pi/(dx*nx)*(nx/2-1)
    wmin = 2*np.pi/(dt*nt)*(-nt/2)
    wmax = 2*np.pi/(dt*nt)*(nt/2-1)
    kaxis = np.linspace(kmin,kmax,nx)
    waxis = np.linspace(wmin,wmax,nt)
    ex0 = np.fft.fftshift(np.fft.fft2(ex0))
    cmap = plt.get_cmap('Blues')
    plt.pcolormesh(kaxis, waxis, np.abs(ex0),cmap=cmap)
    plt.xlabel('$k \lambda_D$')
    plt.ylabel('$\omega/\omega_p$')
    cbar = plt.colorbar()
    cbar.set_label('|$E/E_0$|')
    plt.savefig('tmp/efield_k_omega_run02a.png')
    plt.show()
    #plt.close()

#xt_plot()
main_plot()
