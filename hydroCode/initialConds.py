import numpy as np
import scipy.special as ss
import pylab as plt
## Contains several functions for setting the initial conditions of U,F in 2D hydrocode
def energyFunc(p,gamma,rho,vx,vy):
	# Calculate energy in terms of p, gamma, rho, v
	return p/(gamma-1) + 0.5* rho*(vx**2+vy**2)
	
	
def sod1D(U,F,gamma, rhoL,pL,vxL,vyL, rhoR,pR,vxR,vyR):
	Nx = len(U[0,:,0,0])
	## Left state
	U[:,0:Nx/2,0,0] = rhoL
	U[:,0:Nx/2,0,1] = rhoL*vxL
	U[:,0:Nx/2,0,2] = rhoL*vyL
	U[:,0:Nx/2,0,3] = energyFunc(pL,gamma,rhoL,vxL,vyL)

	F[:,0:Nx/2,0,0] = rhoL*vxL
	F[:,0:Nx/2,0,1] = rhoL*vxL**2 + pL
	F[:,0:Nx/2,0,2] = rhoL*vxL*vyL
	F[:,0:Nx/2,0,3] = (energyFunc(pL,gamma,rhoL,vxL,vyL) + pL)*vxL

	## Right State
	U[:,Nx/2:,0,0] = rhoR
	U[:,Nx/2:,0,1] = rhoR*vxR
	U[:,Nx/2:,0,2] = rhoR*vyR
	U[:,Nx/2:,0,3] = energyFunc(pR,gamma,rhoR,vxR,vyR)

	F[:,Nx/2:,0,0] = rhoR*vxR
	F[:,Nx/2:,0,1] = rhoR*vxR**2 + pR
	F[:,Nx/2:,0,2] = rhoR*vxR*vyR
	F[:,Nx/2:,0,3] = (energyFunc(pR,gamma,rhoR,vxR,vyR) + pR)*vxR
	return U , F

def kelvHelm(U,F,xPoints,yPoints,gamma,p0,rho1,rho2,u1,u2,delta,w0):
	# Initial conditions for Kelvin-Helmholtz instability
	Lx = xPoints[-1] - xPoints[0] 	
	Ly = yPoints[-1] - yPoints[0]
	if np.abs(Lx-Ly) > 10**(-14):
		print 'Need a square domain for this test, taking length value as L'
	L = min(Lx,Ly)
	X, Y = np.meshgrid(xPoints, yPoints)
	
	rhos = rho1 + (rho2-rho1)/2. * (np.tanh((yPoints - L/4.)/delta) - np.tanh((yPoints-3.*L/4.)/delta))
	us   = (u2-u1)/2. * (np.tanh((yPoints - L/4.)/delta) - np.tanh((yPoints - 3.*L/4.)/delta) - 1.) #horizontal vel
	vs   = w0*np.sin(4.*np.pi*xPoints)#vertical vel
	
	#plt.plot(xPoints,vs)
	#plt.show()
	
	U[:,:,0] = rhos[None,:]
	U[:,:,1] = rhos[None,:] * us[None,:]
	U[:,:,2] = rhos[None,:] * vs[:,None]
	U[:,:,3] = energyFunc(p0,gamma,rhos[None,:],us[None,:],vs[:,None])
	
	F[:,:,0] = rhos[None,:] * us[None,:]
	F[:,:,1] = rhos[None,:] * us[None,:]**2 + p0
	F[:,:,2] = rhos[None,:] * us[None,:] * vs[:,None]
	F[:,:,3] = (energyFunc(p0,gamma,rhos[None,:],us[None,:],vs[:,None]) + p0) *us[None,:]
	
	#plt.imshow(U[:,:,0,3],interpolation='gaussian')
	#plt.show()
	return U, F
	
def kelvHelm2(U,F,xPoints,yPoints,gamma,p0,rho0,w0,a,n,vy):
	# Initial conditions for Kelvin-Helmholtz instability
	Lx = xPoints[-1] - xPoints[0] 	
	Ly = yPoints[-1] - yPoints[0]
	Nx = len(xPoints)
	if np.abs(Lx-Ly) > 10**(-14):
		print 'Need a square domain for this test, taking length value as L'
	L = min(Lx,Ly)
	X, Y = np.meshgrid(xPoints, yPoints)
	
	rhos = rho0
	us   = w0*np.exp(-(xPoints[None,:]-Lx/2.)**2/a**2)*np.sin(2*np.pi*n*yPoints[:,None]/L)
	vs   = np.zeros(Nx)
	vs[0:Nx/2] = vy
	vs[Nx/2:] = -vy
	
	#plt.plot(yPoints,rhos)
	#plt.show()
	
	U[:,:,0] = rhos
	U[:,:,1] = rhos * us
	U[:,:,2] = rhos * vs[:,None]
	U[:,:,3] = energyFunc(p0,gamma,rhos,us,vs[:,None])
	
	F[:,:,0] = rhos * us
	F[:,:,1] = rhos * us**2 + p0
	F[:,:,2] = rhos * us* vs[:,None]
	F[:,:,3] = (energyFunc(p0,gamma,rhos,us,vs[:,None]) + p0) *us
	
	#plt.imshow(U[:,:,2],interpolation='gaussian')
	#plt.show()
	return U, F
	
def implosion(U,F,xPoints,yPoints,gamma,p1,p2,rho1,rho2,vx1,vx2,vy1,vy2):
	# Determine conditions above and below diagonal, helps test symmetry about x=y 
	L = xPoints[-1]-xPoints[0] # Assume square zone
	Nx = len(xPoints)
	Ny = len(yPoints)
	rho = np.zeros([Ny,Nx])
	p   = np.zeros([Ny,Nx])
	vx  = np.zeros([Ny,Nx])
	vy  = np.zeros([Ny,Nx])
	
	for i in range(Nx):
		for k in range(Ny):
			if xPoints[i]+yPoints[k] > L/2.:
				rho[k,i] = rho1
				p  [k,i] = p1
				vx [k,i] = vx1
				vy [k,i] = vy1
			else:
				rho[k,i] = rho2
				p  [k,i] = p2
				vx [k,i] = vx2
				vy [k,i] = vy2
	U[:,:,0] = rho
	U[:,:,1] = rho * vx
	U[:,:,2] = rho * vy
	U[:,:,3] = energyFunc(p,gamma,rho,vx,vy)
	
	F[:,:,0] = rho * vx
	F[:,:,1] = rho * vx**2 + p
	F[:,:,2] = rho * vx* vy
	F[:,:,3] = (energyFunc(p,gamma,rho,vx,vy) + p) *vx
	return U, F
	
def bessel(t,U,F,xPoints,gamma,rho,p,vx,vy,xScale,pScale):
	#pBess = np.zeros([len(yPoints),len(xPoints)])
	pBess = (np.cos(2*np.pi*t)+1)*pScale*np.abs(ss.j0(xScale*xPoints))# Pressure at a boundary due to Bessel beam
	if t < 10**(-14):
		# Uniform fluid throughout
		U[:,:,0] = rho
		U[:,:,1] = rho*vx
		U[:,:,2] = rho*vy
		U[:,:,3] = energyFunc(p,gamma,rho,vx,vy)
		F[:,:,0] = rho*vx
		F[:,:,1] = rho*vx**2 + p
		F[:,:,2] = rho*vx*vy
		F[:,:,3] = (energyFunc(p,gamma,rho,vx,vy) + p)*vx

	# Change one boundary to have pressure pBound
	U[0:2,:,3] = energyFunc(pBess,gamma,rho,vx,vy)
	F[0:2,:,1] = rho*vx**2 + pBess
	F[0:2,:,3] = (energyFunc(pBess,gamma,rho,vx,vy) + pBess)*vx

	
	#plt.plot(U[0,:,0,2])
	#plt.show()
	return U, F
