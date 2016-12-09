import numpy as np

## Functionalize boundary conditions for ease of use
## All for 2 ghost cells on each side
## The row directions spans vertical (y), the column spans horizontal (x), but I haven't been totally consistent with that so far
## That is, row direction is direction along which row changes, column direction is that along which column changes


# Fixed BCS in row and column directions
def fixed(U,F,IU,IF):
	#I denotes initial array of that type (U or F)
	# dimensions are row-dim, col-dim, conserved-quantity (mass, x momentum, y momentum, E)
	U[0:2,:,:]  = IU[0:2,:,:]
	F[0:2,:,:]  = IF[0:2,:,:]

	U[-2::,:,:] = IU[-2::,:,:]
	F[-2::,:,:] = IF[-2::,:,:]

	
	U[:,0:2,:]  = IU[:,0:2,:]
	F[:,0:2,:]  = IF[:,0:2,:]

	U[:,-2::,:] = IU[:,-2::,:]
	F[:,-2::,:] = IF[:,-2::,:]
	return U, F
	
# Outflow BCS in row and column directions
def outflow(U,F,IU,IF):
	U[0,:,:]  = U[2,:,:]
	F[0,:,:]  = F[2,:,:]
	U[-1,:,:] = U[-3,:,:]
	F[-1,:,:] = F[-3,:,:]
	
	U[:,0,:]  = U[:,2,:]
	F[:,0,:]  = F[:,2,:]
	U[:,-1,:] = U[:,-3,:]
	F[:,-1,:] = F[:,-3,:]
	
	U[1,:,:]  = U[2,:,:]
	F[1,:,:]  = F[2,:,:]
	U[-2,:,:] = U[-3,:,:]
	F[-2,:,:] = F[-3,:,:]
	
	U[:,1,:]  = U[:,2,:]
	F[:,1,:]  = F[:,2,:]
	U[:,-2,:] = U[:,-3,:]
	F[:,-2,:] = F[:,-3,:]
	return U, F
	
# Periodic in row directions, fixed in column direction
def periodicRow(U,F,IU,IF):
	U[:,0,:]  = IU[:,0,:]
	F[:,0,:]  = IF[:,0,:]
	U[:,-1,:] = IU[:,-1,:]
	F[:,-1,:] = IF[:,-1,:]
	
	U[0,:,:]  = (U[-3,:,:])#+U[:,2,:])/2.
	F[0,:,:]  = (F[-3,:,:])#+F[:,2,:])/2.
	U[-1,:,:] = (U[-3,:,:])#+U[:,2,:])/2.
	F[-1,:,:] = (F[-3,:,:])#+F[:,2,:])/2.
	
	U[:,1,:]  = IU[:,1,:]
	F[:,1,:]  = IF[:,1,:]
	U[:,-2,:] = IU[:,-2,:]
	F[:,-2,:] = IF[:,-2,:]
	
	U[1,:,:]  =(U[-3,:,:])#+U[:,2,:])/2.
	F[1,:,:]  = (F[-3,:,:])#+F[:,2,:])/2.
	U[-2,:,:] = (U[-3,:,:])#+U[:,2,:])/2.
	F[-2,:,:] =(F[-3,:,:])#+F[:,2,:])/2.
	return U, F
