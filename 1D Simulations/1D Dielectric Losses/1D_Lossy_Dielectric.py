import numpy as np
from math import exp,sin,pi
from matplotlib import pyplot as plt

def main():

    # SIMULATION SETTINGS
    nsteps = 400
    grid_size = 140
    grid_center = int(grid_size / 2)
    Ex = np.zeros(grid_size)
    Hy = np.zeros(grid_size)

    # PULSE SETTINGS
    freq = 700e6 # 700MHz
    amp = 1
    c = 3e8
    wavelength = c/(freq)

    # RESOLUTION
    dx = .01 # 10 points per wavelength
    dt = dx/(2*c)

    # ABC STORAGE
    boundary_left = [0,0]
    boundary_right = [0,0]

    # DIELECTRIC PARAMETERS
    eps_0 = 8.854e-12
    epsilon = 4
    sigma = 0.04
    
    # LOAD GEOMETRY
    C1 = np.ones(grid_size)
    C2 = np.ones(grid_size)
    C2 *= 0.5
    C_start = 70

    gamma = dt*sigma/(2*eps_0*epsilon)
    C1[C_start:] = ((1-gamma)/(1+gamma))
    C2[C_start:] = 1/(2*epsilon*(1+gamma))
    
    # FDTD LOOP
    for t in range(1, nsteps + 1):
        
        #UPDATE ELECTRIC FIELD
        for k in range(1, grid_size):
            Ex[k] = C1[k]*Ex[k] + C2[k]*(Hy[k-1] - Hy[k])
            
        # SOURCE
        source = amp*sin(2*pi*freq*dt*t)
        Ex[5] += source

        # ABCs
        Ex[0] = boundary_left.pop(0)
        boundary_left.append(Ex[1])
        Ex[grid_size-1] = boundary_right.pop(0)
        boundary_right.append(Ex[grid_size-2])

        #UPDATE MAGNETIC FIELD
        for k in range(grid_size - 1):
            Hy[k] = Hy[k] + 0.5*(Ex[k] - Ex[k+1])
            
        # PLOTTING
        plt.rcParams['font.size'] = 12
        plt.figure(figsize=(8, 3.5))
        plt.subplot(211)
        plt.title("{:.2e}".format(freq)+'Hz')
        plt.plot(Ex, color='k', linewidth=1)
        plt.ylabel('E$_x$', fontsize='14')
        plt.xticks(np.arange(0, grid_size+1, step=20))
        plt.xlim(0, grid_size)
        plt.yticks(np.arange(-1, 1.2, step=1))
        plt.ylim(-1.2, 1.2)
        plt.text(grid_size-20, 1.25, 't = {}'.format(t),
                 horizontalalignment='left')
        plt.subplot(212)
        plt.plot(Hy, color='k', linewidth=1)
        plt.ylabel('H$_y$', fontsize='14')
        plt.xlabel('cm')
        plt.xticks(np.arange(0, grid_size+1, step=20))
        plt.xlim(0, grid_size)
        plt.yticks(np.arange(-1, 1.2, step=1))
        plt.ylim(-1.2, 1.2)
        plt.subplots_adjust(bottom=0.2, hspace=0.45)
        plt.savefig('png/'+str(t)+'.png')
        plt.close()
        
        

if __name__ == '__main__':
    main()
