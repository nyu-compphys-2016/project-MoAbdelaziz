import numpy as np
import pylab as plt
import scipy.integrate as si

## Calculate the entropy in the system throughout the run check how well it's conserved for different resolutions

#dataName = 'isen16.npy'
dataNames = ['isen16.npy' , 'isen32.npy' ,'isen48.npy' ,'isen64.npy' ,'isen80.npy' ,'isen96.npy', 'isen112.npy','isen128.npy' ,'isen144.npy','isen160.npy']
L1Array = np.zeros(len(dataNames))
NArray  = np.zeros(len(dataNames))
count = -1
for dataName in dataNames:
	count += 1
	rawData = np.load(dataName)[:,:,:,1:] #Load data, throwing out empty slice from bad stacking



	Nx = len(rawData[0,:,0,0])
	Ny = len(rawData[:,0,0,0])
	Nt = len(rawData[0,0,0,:])
	gamma = 1.4 # Used in every test; in the future should make this a variable part of the data
	rho   = rawData[:,:,0,:]
	rho0  = rho[:,:,0]
	p     = (gamma-1) * (rawData[:,:,3,:] - 0.5 * (rawData[:,:,1,:]**2/rho + rawData[:,:,2,:]**2/rho))
	p0    = p[:,:,0]

	# Calculate |s(x,y,t) - s0|
	sDiff = np.abs(1./(gamma-1)*np.log10(p/p0[:,:,None] * (rho/rho0[:,:,None])**(-gamma)))
	#dx = 1.0/Nx
	#dy = 1.0/Ny #unfortunately did not save actual lengths used, but they were 1.0x1.0 for CI and KH cases

	x = np.linspace(0,1,Nx)
	y = np.linspace(0,1,Ny)

	L1 = np.zeros(Nt)
	for t in range(Nt):
		L1[t] = si.simps(si.simps(sDiff[:,:,t], y,axis=0), x,axis=0)
	# The result of this integral will be in grid number units, which should be okay since
	# the row and column grids have each been uniform

	#plt.plot(L1)
	#plt.show()
	L1Array[count] = L1[-1]
	NArray[count]  = Nx
	#print Nx,Ny,L1[-1] 
	
plt.plot(NArray,L1Array,'k-o',lw = 3, ms =10)
plt.tick_params(labelsize=14)
plt.title('Isentropic Wave Entropy Convergence' , fontsize = 24)
plt.xlabel('Number of Cells N in Either Dimension', fontsize =18)
plt.ylabel('L1 Error in Specific Entropy' , fontsize = 18)
plt.xscale('log')
plt.yscale('log')
plt.xlim([16,160])
plt.show()

roi = (slice(1,len(NArray)))
fit = np.polyfit(np.log(NArray[roi]), np.log(L1Array[roi]), 1)[0] 
print fit

