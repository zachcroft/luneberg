# SIMULATION SETTINGS
W, H = 250, 250 # Simulation Dimensions
nsteps = 200


# SOURCE SETTINGS
c = 3e8

source_type = 'sinewave'

source_freq = 1500e6 #300 Thz is a 1 um signal
wavelength = c/source_freq

amplitude = 1 # pulse amplitude

# RESOLUTION
cell_size = .01 # should be 1/10 the wavelength

source_x = 50 #Source x position
source_y = 104 #Source y position

source2_x = 50 #Source x position
source2_y = 143 #Source y position

# DIELECTRIC SETTINGS
conductivity = 1e7
dielec_const = 12

# PLOT SETTINGS
plot_title = "Bent Waveguide, $f$ = "+"{:.2e}".format(source_freq)+"Hz"
plot_type = 'flat'
color_map = 'RdBu'
#color_map = 'jet'
show_dielectric = True #Note:increases runtime
dielectric_transparency = 0.05
