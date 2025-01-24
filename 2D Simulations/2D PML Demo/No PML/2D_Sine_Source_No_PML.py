import settings
import numpy as np
import mpl_toolkits.mplot3d.axes3d
from math import sin,cos,pi,sqrt,exp,atan2
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

# DIELECTRIC SHAPES
def draw_rect(material,width,height,x_pos,y_pos):
    if material == 'vacuum':
        for x in range(0, height):
            for y in range(0, width):
                C1[x+y_pos, y+x_pos] = 1
                C2[x+y_pos, y+x_pos] = 1
    elif material == 'dielectric':
        for x in range(0, height):
            for y in range(0, width):
                C1[x+y_pos, y+x_pos] = 1 / (epsr + (sigma * dt / epsz))
                C2[x+y_pos, y+x_pos] = (sigma * dt / epsz)

# PLOTTING
def plot_e_color(fig,ax,data,timestep,X,Y):
    z_min, z_max = -np.abs(data).max(), np.abs(data).max()
    #fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, data, cmap=settings.color_map, \
                      vmin=-1, vmax=1,shading='auto')
    if settings.show_dielectric == True:
        ax.pcolormesh(X, Y, np.copy(C1), cmap='binary', \
                      alpha=settings.dielectric_transparency, \
                      vmin=0, vmax=1,shading='auto')
    ax.set_title(settings.plot_title)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.set_xlabel('cm')
    fig.colorbar(c, ax=ax)

def plot_e_field(ax,data,timestep,X,Y):
    ax.set_zlim(-1, 1)
    ax.view_init(elev=15., azim=25)
    ax.plot_surface(Y, X, data, rstride=1, cstride=1,
                    cmap='RdBu',
                    vmin=-1, vmax=1, linewidth=.25)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r' $E_{Z}$', rotation=90, labelpad=10,
                  fontsize=14)
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    ax.set_xticks(np.arange(0, sim_width+20, step=20))
    ax.set_yticks(np.arange(0, sim_height+20, step=20))
    ax.set_zticks([-0.5, 0, 0.5])
    ax.text2D(0.8, 0.8, "t = {}".format(timestep),
              transform=ax.transAxes)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = \
                         ax.zaxis.pane.fill = False
    plt.gca().patch.set_facecolor('white')
    ax.dist = 11

# DATA HANDLING
def field_data_in(filename):
    file2 = open(filename,"r")
    file2.readline() #Simulation Header
    file2.readline() #Column labels Ez,Dz,Hx,Hy,iHx,iHy,Ez_inc,Hx_inc,Iz
    for j in range(0, sim_width):
        for i in range(0, sim_height):
            line = file2.readline()
            B = line.split('\t')
            Ez[i,j]=B[0]
            Dz[i,j]=B[1]
            Hx[i,j]=B[2]
            Hy[i,j]=B[3]
            iHx[i,j]=B[4]
            iHy[i,j]=B[5]
            Ez_inc[j]=B[6]
            Hx_inc[j]=B[7]
            iz[i,j]=B[8]         
    file2.close()

def field_data_out(filename,time_step):
    file1 = open(filename,"a")
    file1.write("Simulation at t="+str(time_step)+'\n')
    file1.write("Ez"+'\t'+"Dz"+'\t'+"Hx"+'\t'+"Hy"+'\t'+"iHx"+'\t'+"iHy"+'\t'+"Ez_inc"+'\t'+"Hx_inc"+'\t'+"iz"'\n')
    for j in range(0, sim_width):
        for i in range(0, sim_height):
            file1.write(str(Ez[i,j])+'\t'+str(Dz[i,j])+'\t'+str(Hx[i,j])+'\t'+str(Hy[i,j])+'\t'+str(iHx[i,j])+'\t'+str(iHy[i,j])+'\t'+str(Ez_inc[j])+'\t'+str(Hx_inc[j])+'\t'+str(iz[i,j])+'\n')
    file1.close()
    

# LOAD SIMULATION SETTINGS
sim_width = settings.W
sim_height = settings.H

nsteps = settings.nsteps
freq = settings.source_freq

source_x = settings.source_x
source_y = settings.source_y

# RESOLUTION
c = 3e8
dx = settings.cell_size # Cell size
dt = 0.5*dx/c # Time step size

# PULSE PARAMETERS
t0 = 40
spread = 8
amp = settings.amplitude

# INITIALIZE FIELDS
Ez = np.zeros((sim_height, sim_width))
Dz = np.zeros((sim_height, sim_width))
Hx = np.zeros((sim_height, sim_width))
Hy = np.zeros((sim_height, sim_width))
iHx = np.zeros((sim_height, sim_width))
iHy = np.zeros((sim_height, sim_width))
Ez_inc = np.zeros(sim_width)
Hx_inc = np.zeros(sim_width)
iz = np.zeros((sim_height, sim_width))
C1 = np.ones((sim_height, sim_width))
C2 = np.zeros((sim_height, sim_width))

# DIELECTRIC PARAMETERS
eps_0 = 8.854e-12
epsr = settings.dielec_const
sigma = settings.conductivity

# LOAD DIELECTRIC GEOMETRY
draw_rect('vacuum',sim_width,sim_height,0,0)

# MAIN FDTD LOOP
def main():
    #field_data_in("fields_0_200.txt")
    t = 1
    for iter in range(1,nsteps+1):
        
        # UPDATE DIELECTRIC FIELD
        for j in range(1, sim_width):
            for i in range(1, sim_height):
                Dz[i,j] = Dz[i,j] + 0.5 * (Hy[i,j] - Hy[i-1,j] -
                                           Hx[i,j] + Hx[i,j-1])
        # SPECIFY SOURCE
        if settings.source_type == 'sinewave':
            source = amp*sin(2*pi*freq*dt*t)
        elif settings.source_type == 'gaussian':
            source = exp(-0.5*((t0 - t)/spread)**2)

        Dz[source_x,source_y] = source

        # UPDATE ELECTRIC FIELD
        for j in range(1, sim_width):
            for i in range(1, sim_height):
                Ez[i,j] = C1[i,j]*Dz[i,j]

        # UPDATE Hx FIELD
        for j in range(sim_width-1):
            for i in range(sim_height-1):
                Hx[i, j] = Hx[i,j] + 0.5*(Ez[i,j] - Ez[i,j+1])
        # UPDATE Hy FIELD
        for j in range(sim_width-1):
            for i in range(sim_height-1):
                Hy[i,j] = Hy[i,j] + 0.5*(Ez[i+1,j] - Ez[i,j])
         
        # PLOTTING
        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linestyle'] = 'dotted'
        if settings.plot_type == 'flat':
            fig = plt.figure(figsize=(10, 8))
        elif settings.plot_type == '3D':
            fig = plt.figure(figsize=(8, 8))
        X, Y = np.meshgrid(range(sim_width), range(sim_height))
        if settings.plot_type == 'flat':
            ax = fig.add_subplot(1, 1, 1)
            plot_e_color(fig,ax, np.copy(Ez),t,X,Y)
        elif settings.plot_type == '3D':
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            plot_e_field(ax, np.copy(Ez),t,X,Y)
        plt.savefig('png/'+str(t)+'.png')
        plt.close()
        print('Time step: '+str(t))
        t+=1
    field_data_out("fields_0_200.txt",t)

if __name__ == '__main__':
    main()
    



