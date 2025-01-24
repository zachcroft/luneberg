import numpy as np
from math import exp
from matplotlib import pyplot as plt

def main():

    # SETTINGS
    nsteps = 400
    grid_size = 200
    grid_center = int(grid_size / 2)
    Ex = np.zeros(grid_size)
    Hy = np.zeros(grid_size)

    # GAUSSIAN PULSE
    t0 = 40
    std_dev = 12

    # ABC STORAGE
    boundary_left = [0, 0]
    boundary_right = [0, 0]
    
    # LOAD DIELECTRIC GEOMETRY
    C = np.ones(grid_size)
    C = 0.5 * C
    dielectric_left = 100
    epsilon = 4
    C[dielectric_left:] = 0.5 / epsilon

    # FDTD LOOP
    for t in range(1,nsteps+1):
        
        # UPDATE ELECTRIC FIELD
        for k in range(1, grid_size):
            Ex[k] = Ex[k] + C[k]*(Hy[k-1] - Hy[k])
            
        # SOURCE
        source = exp(-0.5*((t0-t)/std_dev)**2)
        Ex[5] += source

        # Absorbing Boundary Conditions
        Ex[0] = boundary_left.pop(0)
        boundary_left.append(Ex[1])
        Ex[grid_size-1] = boundary_right.pop(0)
        boundary_right.append(Ex[grid_size-2])
        
        # UPDATE MAGNETIC FIELD
        for k in range(grid_size - 1):
            Hy[k] = Hy[k] + 0.5*(Ex[k] - Ex[k+1])
            
        # PLOTTING
        plt.rcParams['font.size'] = 12
        plt.figure(figsize=(8, 3.5))
        plt.subplot(211)
        plt.plot(Ex, color='k', linewidth=1)
        plt.ylabel('E$_x$', fontsize='14')
        plt.xticks(np.arange(0, 201, step=20))
        plt.xlim(0, 200)
        plt.yticks(np.arange(-1, 1.2, step=1))
        plt.ylim(-1.2, 1.2)
        plt.text(180, 1.25, 't = {}'.format(t),
                 horizontalalignment='left')
        plt.subplot(212)
        plt.plot(Hy, color='k', linewidth=1)
        plt.ylabel('H$_y$', fontsize='14')
        plt.xlabel('Position')
        plt.xticks(np.arange(0, 201, step=20))
        plt.xlim(0, 200)
        plt.yticks(np.arange(-1, 1.2, step=1))
        plt.ylim(-1.2, 1.2)
        plt.subplots_adjust(bottom=0.2, hspace=0.45)
        plt.savefig('png/'+str(t)+'.png')
        plt.close()
        plt.show()
        
if __name__ == '__main__':
    main()
