# Here some parameters inspired by experimental articles 
import numpy as np 
from Metamaterial import SRR_metamaterial

L = 13.459 * 10 ** -9                   # Self-inductance
C = 470 * 10 ** -10                     # Capacitance
R = 0.002                               # Resistance
# R = 0.00465
omega_0 = 1 / np.sqrt(L * C)            # Resonance frequency
omega_mu = 6.24627e6*2*np.pi  			# Resonance frequency for effective mu
omega_sp_alpha = 6.41727e6*2*np.pi		# Resonance frequency for spherical polarizability

Omega =  np.linspace(omega_0*0.9, omega_0*1.1, 1000) # Frequency range
H_0z = 1                                # Amplitude of magnetic field
mu_0 = 4 * np.pi * 10 ** -7             # Magnetic constant

dz = 15 * 10 ** -3                      # Length of cell
dy = dz
dx = dz

Radius = 4.935 * 10 ** -3               # Mean radius of rings
W = 0.7 * 0.15 * dz * 0                 # Width of strip

Sigma = -0.06                           # Lattice constant

SRR = SRR_metamaterial(R, L, C, Radius, (dz, dy, dx), W)
