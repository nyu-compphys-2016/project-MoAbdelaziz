import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation

dataName = 'bessel256.npy'
rawData     = np.load(dataName)[:,:,:,1:] # Load data, throwing out empty slice from bad stacking
#value    = 'Density' # Choose quantity to plot: Density, Pressure, Velocity
value = 'Density'
Nx = len(rawData[0,:,0,0])
Ny = len(rawData[:,0,0,0])
Nt = len(rawData[0,0,0,:])
colormap = cm.inferno   # Choose colormap to use
tArray = 100*np.arange((Nt-1)/100)
t        = 0   # Choose timestep to plot for
print Nx,Ny
for t in tArray:
	print 'Running for t = ',t


	title    = 'Colormap of '+value+' for Bessel Beam'
	tickSize  = 14
	labelSize = 18
	titleSize = 24

	## Plot colormap at some time
	fig = plt.figure()
	ax  = fig.gca()

	if value == 'Density':
		data = rawData[:,:,0,t]
	if value == 'Velocity':
		Vx = rawData[:,:,2,t]/rawData[:,:,0,t] # x velocity
		Vy = rawData[:,:,1,t]/rawData[:,:,0,t] # y velocity (seem reversed cause of poor definitions in earlier code
		data = np.sqrt(Vx*Vx+Vy*Vy) #total speed
	#	print data
		X, Y =  np.linspace(-0.5,0.5,Nx), np.linspace(0,5,Ny) ## CHANGE THIS MANUALLY FOR DIFFERENT SYSTEMS so far

	
	if value =='Density':
		plt.imshow(data, origin = 'lower',cmap = colormap)
	if value == 'Velocity':
		plt.streamplot(X, Y, Vx, Vy ,color='k', linewidth=2)
	plt.tick_params(labelsize=tickSize)
	plt.title(title, fontsize = titleSize)
	if value == 'Density':
		plt.xlabel('X Cell Number', fontsize = labelSize)
		plt.ylabel('Y Cell Number', fontsize = labelSize)
		plt.xlim([3,Nx-2])
		plt.ylim([3,Ny-2])
		m = cm.ScalarMappable(cmap= colormap)
		m.set_array(data)
		plt.colorbar(m)
	if value == 'Velocity':
		plt.xlabel('X Position', fontsize = labelSize)
		plt.ylabel('Y Position', fontsize = labelSize)
		plt.xlim([min(X),max(X)])
		plt.ylim([min(Y),max(Y)])
		#ax.set_aspect('equal')

	dataNameShort = dataName[0:-4]
	plt.savefig(dataNameShort+'t'+str(t)+value+'.png')

	#plt.show()
