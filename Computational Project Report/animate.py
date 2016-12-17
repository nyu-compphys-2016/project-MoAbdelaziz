import numpy as np
import pylab as plt
import eulerExact as ee
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
import initialConds
import boundaryConds
import os
import time

## Note: if X and Y velocities seem reversed, it's because I was inconsistent with rows/columbns equating to x/y.

dataName = 'ci128.npy'
rawData     = np.load(dataName)[:,:,:,1:] # Load data, throwing out empty slice from bad stacking
value    = 'Density' # Choose quantity to plot: Density, Pressure, Velocity
colormap = cm.inferno
#value = 'Pressure'
#value = 'X Velocity'
#value = 'Y Velocity'
Nx = len(rawData[0,:,0,0])
Ny = len(rawData[:,0,0,0])
Nt = len(rawData[0,0,0,:])
gamma =1.4

fig = plt.figure()
ax = fig.gca()
#m = cm.ScalarMappable(cmap=cm.jet)
#m.set_array(rawData[2:-2,2:-2,0,0])
#plt.colorbar(m)

frameStep = 5
def animate(i):
	ax.clear()
	if value == 'Density':
		surf = plt.imshow(rawData[2:-2,2:-2,0,i] , origin = 'lower',interpolation ='none', cmap = colormap, vmin = 0.3, vmax = 1.3)   # update the data
	if value == 'Pressure':
		rho   = rawData[:,:,0,i]
		surf= plt.imshow((gamma-1) * (rawData[:,:,3,i] - 0.5 * (rawData[:,:,1,i]**2/rho + rawData[:,:,2,i]**2/rho)))
	if value == 'X Velocity':
		rho   = rawData[:,:,0,i]
		surf = plt.imshow(rawData[:,:,2,i]/rho)
	if value == 'Y Velocity':
		rho   = rawData[:,:,0,i]
		surf = plt.imshow(rawData[:,:,1,i]/rho)
	plt.xlabel('X Cell' , fontsize = 18)
	plt.ylabel('Y Cell', fontsize =18)
	plt.xlim([2,Nx-5])
	plt.ylim([2,Ny-5])
	plt.title(value+' Colormap for a Corner Implosion', fontsize = 24)
	#ax.clear() # Seems necessary to prevent data overlapping between frames
	#ax.set_zlim([rhoR, rhoL])
	
#	surf = plt.streamplot(X, Y, UWhole[:,:,1,i]*((tMax)/Nt)/UWhole[:,:,0,i] ,UWhole[:,:,2,i]*((tMax)/Nt)/UWhole[:,:,0,i],          # data
#               color='blue',         # array that determines the colour
#               cmap=cm.seismic,        # colour map
#               linewidth=2,         # line thickness
#               arrowstyle='->',     # arrow style
#               arrowsize=1.5)       # arrow size
#               
               
	if (i/frameStep) % 10 ==0:
		print 'Animating Frame', i/frameStep , '/' , Nt/frameStep
		
	return surf, 

m = cm.ScalarMappable(cmap=colormap)
if value == 'Density':
	m.set_array(rawData[2:-2,2:-2,0,:])
if value == 'Pressure':
	rho   = rawData[:,:,0,-1]
	m.set_array((gamma-1) * (rawData[:,:,3,-1] - 0.5 * (rawData[:,:,1,-1]**2/rho + rawData[:,:,2,-1]**2/rho)))
if value == 'X Velocity':
	rho   = rawData[:,:,0,-1]
	m.set_array(rawData[:,:,2,-1]/rho)
if value == 'Y Velocity':
	rho   = rawData[:,:,0,-1]
	m.set_array(rawData[:,:,1,-1]/rho)
plt.colorbar(m)
# Init only required for blitting to give a clean slate.
def init():
	ax.plot_surface([],[],[], rstride=5, cstride=5,cmap=cm.coolwarm)  
	return surf,

ani = animation.FuncAnimation(fig, animate, np.arange(1,Nt,frameStep),  interval=25, blit=False)
dpi = 100
dataNameShort = dataName[0:-4]
ani.save(dataNameShort+value+str(dpi)+'.mp4' , dpi = dpi)

plt.show(fig)

