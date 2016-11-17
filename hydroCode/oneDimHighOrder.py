import numpy as np
import pylab as plt

## FUNCTION DEFINITIONS

def LFunc(U,F,dx,n):
	# Calculate L(U) given a U[:,n,0:2] for the conserved variable array at time n, all positions
	FHLL = FHLLFunc(U,F,n) 
	
	L = -(1/dx) * (FHLL[1::,n,:] - FHLL[0:-1,n,:])
	return L
	
def FHLLFunc(U,F,n):
	# Calculate HLL Flux as an array FHLL[:,0:2] at all interfaces (one less entry than cell positions)
	
	rho = U[:,n,0]
	v = U[:,n,1]/U[:,n,0]
	p = F[:,n,1] - U[:,n,1]**2 / U[:,n,0]
	lambdaPlus, lambdaMinus = lambdaFunc(v,gamma,p,rho)
	alphaPlus, alphaMinus = alphaFunc(lambdaPlus,lambdaMinus)
	
	FHLL[:,n,:] = (alphaPlus[:,None]*F[0:-1,n,:] + alphaMinus[:,None]*F[1::,n,:] - alphaPlus[:,None]*alphaMinus[:,None] *(U[1::,n,:] - U[0:-1,n,:]))/(alphaPlus[:,None] + alphaMinus[:,None])
	return FHLL
	
def lambdaFunc(v,gamma,p,rho):
	# Calculate lambda as an array at all positions lambdas[:] at some time 
	# v[:] at all positions
	# gamma constant
	# p[:] at all positions	
	# rho[:] at all positions
	sound = np.sqrt(gamma*p/rho)
	lambdaPlus = v + sound
	lambdaMinus = v - sound
	return lambdaPlus, lambdaMinus
	
def alphaFunc(lambdaPlus,lambdaMinus):
	# Find alphas at all interfaces (one less entry than cell positions)
	alphaPlus = np.maximum(0,lambdaPlus[0:-1],lambdaPlus[1::])
	alphaMinus = np.maximum(0,-lambdaMinus[0:-1],-lambdaMinus[1::])
	return alphaPlus,alphaMinus

def energyFunc(p,gamma,rho,v):
	# Calculate energy in terms of p, gamma, rho, v
	return p/(gamma-1) + 0.5* rho*v**2

def fluxUpdate(U,F,n):
	# Calculate updated values of F based on a new U
	newRho  = U[1:-1,n+1,0]
	newRhoV = U[1:-1,n+1,1]
	newV    = newRhoV/newRho
	newE    = U[1:-1,n+1,2]
	newP    = (gamma-1) * (newE - 0.5*newRho*newV**2)
	
	F[1:-1,n+1,0] = newRho*newV
	F[1:-1,n+1,1] = newRho*newV**2 + newP
	F[1:-1,n+1,2] = (newE+newP)*newV
	return F
## COMPUTATION PARAMETERS
tMin    = 0.
tMax    = 0.2
Nt      = 1000
dt      = (tMax-tMin)/Nt
tPoints = np.linspace(tMin,tMax,Nt)

xMin    = 0.
xMax    = 1.
Nx      = 1000
dx      = (xMax - xMin)/Nx

xPoints = np.linspace(xMin,xMax,Nx)
## INITIAL CONDITIONS
gamma = 1.4 # Adiabatic index
pL = 1.0 # Pressure on the left
rhoL = 1.0 # Mass dens on the left
vL = 0.0 # Velocity on the left

pR = 0.125 # Right
rhoR = 0.1
vR = 0.0

U = np.zeros([Nx,Nt,3])
F = np.zeros([Nx,Nt,3])
FHLL = np.zeros([Nx-1,Nt,3])
U1 = np.zeros(U.shape)
U2 = np.zeros(U.shape)
# Left side
U[0:Nx/2,0,0] = rhoL
U[0:Nx/2,0,1] = rhoL*vL
U[0:Nx/2,0,2] = energyFunc(pL,gamma,rhoL,vL)
F[0:Nx/2,0,0] = rhoL*vL
F[0:Nx/2,0,1] = rhoL*vL**2 + pL
F[0:Nx/2,0,2] = (energyFunc(pL,gamma,rhoL,vL) + pL)*vL
# Right side
U[Nx/2::,0,0] = rhoR
U[Nx/2::,0,1] = rhoR*vR
U[Nx/2::,0,2] = energyFunc(pR,gamma,rhoR,vR)
F[Nx/2::,0,0] = rhoR*vR
F[Nx/2::,0,1] = rhoR*vR**2 + pR
F[Nx/2::,0,2] = (energyFunc(pR,gamma,rhoR,vR) + pR)*vR
U1 = U.copy() # Need to maintain BCS for U1 and U2 as well
U2 = U.copy()
F1 = F.copy()
F2 = F.copy()
## MAIN LOOP
count = 0 # Track loop number
for n in range(Nt-1):
	count +=1 
	#RK3 Updating
	U1[1:-1,n,:] = U[1:-1,n,:] + dt*LFunc(U,F,dx,n)
	F1 = fluxUpdate(U1,F,n-1)
	U2[1:-1,n,:] = (3./4.)*U[1:-1,n,:] + (1./4.)*U1[1:-1,n,:] + (1./4.)*dt*LFunc(U1,F1,dx,n)
	F2 = fluxUpdate(U2,F1,n-1)
	U[1:-1,n+1,:] = (1./3.)*U[1:-1,n,:] + (2./3.)*U2[1:-1,n,:] + (2./3.)*dt*LFunc(U2,F2,dx,n)
	# Update all fluxes based on this update
	F = fluxUpdate(U,F,n)
	# Reinforce BCS (return ghost cells to initial conditions) (Need two ghost cells on each side for 2nd order spatial)
	U[0:2,n+1,:] = U[0,0,:]
	F[0:2,n+1,:] = F[0,0,:]
	U[-2::,n+1,:] = U[-1,0,:]
	F[-2::,n+1,:] = F[-1,0,:]
	U1 = U.copy()
	U2 = U.copy()
	F1 = F.copy()
	F2 = F.copy()
	if count % 100 == 0:
		print count*100/Nt , '% Done'
plt.figure()
plt.plot(xPoints,U[:,-1,0], label = 'Sod Test')
plt.ylim([0,rhoL*1.1])
plt.legend()
plt.show()

