# SIMULATION SETTINGS
H = 300 #Simulation Height
W = 500 #Simulation Width
nsteps = 100
cell_size = .01 # Cell size -- should be 1/10 the wavelength

# SOURCE SETTINGS
source_type = 'sinewave'
source_freq = 700e6 #300 Thz is a 1 um signal
wavelength = 1e9*3e8/source_freq
amplitude = 1 # pulse amplitude
source_x = 79 #Source x position
source_y = 150 #Source y position

# DIELECTRIC SETTINGS
conductivity = 0
dielec_const = 12


# PLOT SETTINGS
plot_title = "Luneburg Lens"
plot_type = 'flat'
color_map = 'RdBu'
show_dielectric = True #Note:increases runtime
dielectric_transparency = 0.05
