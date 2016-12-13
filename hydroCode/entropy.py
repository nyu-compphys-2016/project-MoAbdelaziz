import numpy as np
import pylab as plt

## Calculate the entropy in the system throughout the run check how well it's conserved for different resolutions

dataName = 'ci128.npy'
rawData = np.load(dataName)[:,:,1,:] #Load data, throwing out empty slice from bad stacking



Nx = len(rawData[0,:,0,0])
Ny = len(rawData[:,0,0,0])

gamma = 1.4 # Used in every test; in the future should make this a variable part of the data
p     = 
rho   =
p0    = 
rho0  =

