
c = 3e8

# SIMULATION SETTINGS
W, H = 200, 200 # Simulation Dimensions
nsteps = 200


# SOURCE SETTINGS
source_type = 'sinewave'

source_freq = 1500e6 #300 Thz is a 1 um signal
wavelength = c/source_freq

amplitude = 2 # pulse amplitude

# RESOLUTION
cell_size = .01 # should be 1/10 the wavelength

source_x = 100 #Source x position
source_y = 100 #Source y position

# DIELECTRIC SETTINGS
conductivity = 0
dielec_const = 12

# PLOT SETTINGS
plot_title = "PML, $f$ = "+"{:.2e}".format(source_freq)+"Hz"
plot_type = 'flat'
color_map = 'RdBu'
#color_map = 'jet'
show_dielectric = False #Note:increases runtime
dielectric_transparency = 0.05
