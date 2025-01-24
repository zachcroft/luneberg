import settings
import numpy as np
import mpl_toolkits.mplot3d.axes3d
from math import sin,cos,pi,sqrt,exp,atan2
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

# DIELECTRIC SHAPES
def draw_rect(material,espr,width,height,x_pos,y_pos):
    if material == 'vacuum':
        for x in range(0, height):
            for y in range(0, width):
                C1[x+y_pos, y+x_pos] = 1
                C2[x+y_pos, y+x_pos] = 1
    elif material == 'dielectric':
        for x in range(0, height):
            for y in range(0, width):
                C1[x+y_pos,y+x_pos] = 1 / (epsr+(sigma*dt/eps_0))
                C2[x+y_pos,y+x_pos] = (sigma*dt/eps_0)

def draw_circ(material,espr,radius,x_pos,y_pos):
    a = 7
    b = sim_height - a - 1
    a = 7
    b = sim_width - a - 1
    for j in range(a, b):
        for i in range(a, b):
            x = (y_pos - i)
            y = (x_pos - j)
            dist = sqrt(x**2 + y**2)
            if dist <= radius:
                if material == 'vacuum':
                    C1[i,j] = 1
                    C2[i,j] = 1
                elif material == 'dielectric':
                    C1[i,j] = 1/(epsr+(sigma*dt/eps_0))
                    C2[i,j] = (sigma*dt/eps_0)

# DIELECTRIC CONFIGURATIONS
def luneberg(R):
    for r in range(R,0,-1):
        draw_circ('dielectric',2-(r/R)**2,r,125,125)

# PLOTTING
def plot_e_color(fig,ax,data,timestep,X,Y):
    z_min, z_max = -np.abs(data).max(), np.abs(data).max()
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
    #3d Plot of E field at a single time step
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

# SOURCE PARAMETERS
t0 = 40
std_dev = 8
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
draw_rect('dielectric',1,sim_width,sim_height,0,0)
luneberg(50)

# PML PARAMETERS
pml_width = 8
a = pml_width - 1
b = sim_height - a - 1
boundary_left = [0, 0]
boundary_right = [0, 0]

phi_x1 = np.ones(sim_height)
phi_x2 = np.ones(sim_height)
omega_x = np.zeros(sim_height)
psi_x1 = np.ones(sim_height)
psi_x2 = np.ones(sim_height)

phi_y1 = np.ones(sim_width)
phi_y2 = np.ones(sim_width)
omega_y = np.zeros(sim_width)
psi_y1 = np.ones(sim_width)
psi_y2 = np.ones(sim_width)

for n in range(pml_width):
    d = pml_width - n
    loss = d/pml_width
    pml_factor = 0.33*loss**3  # increases as it goes in PML
    
    phi_x1[n] = 1/(1+pml_factor)
    phi_x1[sim_height-1-n] = 1/(1+pml_factor)
    phi_x2[n] = (1-pml_factor)/(1+pml_factor)
    phi_x2[sim_height-1-n] = (1-pml_factor)/(1+pml_factor)
    
    phi_y1[n] = 1/(1+pml_factor)
    phi_y1[sim_width-1-n] = 1/(1+pml_factor)
    phi_y2[n] = (1-pml_factor)/(1+pml_factor)
    phi_y2[sim_width-1-n] = (1-pml_factor)/(1+pml_factor)
    
    loss = (d-0.5)/pml_width
    pml_factor = 0.33*loss**3  # increases as it goes in PML
    
    omega_x[n] = pml_factor
    omega_x[sim_height-2-n] = pml_factor
    psi_x1[n] = 1/(1+pml_factor)
    psi_x1[sim_height-2-n] = 1/(1+pml_factor)
    psi_x2[n] = (1-pml_factor)/(1+pml_factor)
    psi_x2[sim_height-2-n] = (1-pml_factor)/(1+pml_factor)
    
    omega_y[n] = pml_factor
    omega_y[sim_width-2-n] = pml_factor
    psi_y1[n] = 1/(1+pml_factor)
    psi_y1[sim_width-2-n] = 1 / (1+pml_factor)
    psi_y2[n] = (1-pml_factor)/(1 + pml_factor)
    psi_y2[sim_width-2-n] = (1-pml_factor)/(1+pml_factor)
    
# MAIN FDTD LOOP
def main():
    #field_data_in("fields_0_200.txt") # Load saved timesteps 
    t = 1
    for iter in range(1,nsteps+1):
        
        # INCIDENT ELECTRIC FIELD
        for j in range(1, sim_width):
            Ez_inc[j] = Ez_inc[j] + 0.5*(Hx_inc[j-1] - Hx_inc[j])
            
        # ABSORBING BOUNDARY CONDITIONS
        Ez_inc[0] = boundary_left.pop(0)
        boundary_left.append(Ez_inc[1])
        Ez_inc[sim_width - 1] = boundary_right.pop(0)
        boundary_right.append(Ez_inc[sim_width - 2])
        
        # UPDATE DIELECTRIC FIELD
        for j in range(1, sim_width):
            for i in range(1, sim_height):
                Dz[i, j] = phi_x2[i]*phi_y2[j]*Dz[i, j]+phi_x1[i]*phi_y1[j]*\
                           0.5*(Hy[i,j]-Hy[i-1,j] - Hx[i,j]+Hx[i,j-1])
        # SPECIFY SOURCE
        if settings.source_type == 'sinewave':
            source = amp*sin(2*pi*freq*dt*t)
        elif settings.source_type == 'gaussian':
            source = exp(-0.5*((t0 - t)/std_dev)**2)
        
        Dz[source_y, source_x] = source
      

        # INCIDENT DIELECTRIC FIELD
        for i in range(a, b+1):
            Dz[i, a] = Dz[i,a] + 0.5*Hx_inc[a-1]
            Dz[i, b] = Dz[i,b] - 0.5*Hx_inc[b]
        
        Ez = C1*Dz 

        # INCIDENT H FIELD
        for j in range(0, sim_width-1):
            Hx_inc[j] = Hx_inc[j] + 0.5*(Ez_inc[j] - Ez_inc[j+1])
        
        # UPDATE Hx FIELD
        for j in range(sim_width-1):
            for i in range(sim_height-1):
                curl_E = Ez[i,j] - Ez[i,j+1]
                iHx[i,j] = iHx[i,j] + curl_E
                Hx[i,j] = psi_y2[j]*Hx[i,j] + psi_y1[j]*\
                           (0.5*curl_E + omega_x[i]*iHx[i, j])
        # INCIDENT Hx FIELD
        for i in range(a, b+1):
            Hx[i,a-1] = Hx[i,a-1] + 0.5*Ez_inc[a]
            Hx[i,b] = Hx[i,b] - 0.5*Ez_inc[b]

        # UPDATE Hy FIELD
        for j in range(0, sim_width-1):
            for i in range(0, sim_height-1):
                curl_E = Ez[i,j] - Ez[i+1,j]
                iHy[i, j] = iHy[i,j] + curl_E
                Hy[i, j] = psi_x2[i]*Hy[i, j] - psi_x1[i]*\
                           (0.5*curl_E + omega_y[j]*iHy[i,j])
        # INCIDENT Hy FIELD
        for j in range(a,b+1):
            Hy[a-1,j] = Hy[a-1,j] - 0.5*Ez_inc[j]
            Hy[b,j] = Hy[b,j] + 0.5*Ez_inc[j]
            
        # PLOTTING
        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linestyle'] = 'dotted'
        if settings.plot_type == 'flat':
            fig = plt.figure(figsize=(10, 8))
        elif settings.plot_type == '3D':
            fig = plt.figure(figsize=(8, 8))
        X, Y = np.meshgrid(range(sim_width), range(sim_height))
        ax = fig.add_subplot(1, 1, 1)
        if settings.plot_type == 'flat':
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
    



