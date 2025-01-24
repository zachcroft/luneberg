# SIMULATION SETTINGS
W, H = 300, 250 # Simulation Dimensions
nsteps = 200


# SOURCE SETTINGS
c = 3e8

source_type = 'sinewave'

source_freq = 700e6 
wavelength = c/source_freq

amplitude = 3 # pulse amplitude

# RESOLUTION
cell_size = .01 # should be 1/10 the wavelength

source_x = 74 #Source x position
source_y = 125 #Source y position


# DIELECTRIC SETTINGS
conductivity = 0
dielec_const = 12

# PLOT SETTINGS
plot_title = " "
plot_type = 'flat'
color_map = 'RdBu'
#color_map = 'jet'
show_dielectric = True #Note:increases runtime
dielectric_transparency = 0.05
