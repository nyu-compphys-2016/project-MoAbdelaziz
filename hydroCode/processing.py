import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation

dataName = 'bessel256.npy'
rawData     = np.load(dataName)[:,:,:,1:] # Load data, throwing out empty slice from bad stacking


Nx = len(rawData[0,:,0,0])
Ny = len(rawData[:,0,0,0])
Nt = len(rawData[0,0,0,:])

tArray = 100*np.arange((Nt-1)/100)
t        = 0   # Choose timestep to plot for

for t in tArray:
	print 'Running for t = ',t
	value    = 'Density' # Choose quantity to plot: density, pressure, velocity
	colormap = cm.jet    # Choose colormap to use
	title    = 'Colormap of '+value+' for Bessel Beam'
	tickSize  = 14
	labelSize = 18
	titleSize = 24

	## Plot colormap at some time
	fig = plt.figure()
	ax  = fig.gca()

	if value == 'Density':
		data = rawData[:,:,0,t]

	plt.imshow(data, origin = 'lower',cmap = colormap)
	plt.tick_params(labelsize=tickSize)
	plt.title(title, fontsize = titleSize)
	plt.xlabel('X Cell Number', fontsize = labelSize)
	plt.ylabel('Y Cell Number', fontsize = labelSize)
	plt.xlim([3,Nx-2])
	plt.ylim([3,Ny-2])

	m = cm.ScalarMappable(cmap= colormap)
	m.set_array(data)
	plt.colorbar(m)


	dataNameShort = dataName[0:-4]
	plt.savefig(dataNameShort+'t'+str(t)+value+'.png')

	#plt.show()
