import numpy as np
import pylab as plt
import eulerExact as ee

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)
    
    
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
Nt      = 12000
dt      = (tMax-tMin)/Nt
tPoints = np.linspace(tMin,tMax,Nt)

xMin    = 0.
xMax    = 1
Nx      = 12000
dx      = (xMax - xMin)/Nx

#xPoints = np.linspace(xMin,xMax,Nx)
xPoints = xMin + (xMax-xMin)/float(Nx) * (np.arange(Nx) + 0.5)
## INITIAL CONDITIONS
gamma = 1.4 # Adiabatic index
pL = 1. # Pressure on the left
rhoL = 1. # Mass dens on the left
vL = 0.0 # Velocity on the left

pR = .125 # Right
rhoR = .1
vR = 0.0

U = np.zeros([Nx,Nt,3])
F = np.zeros([Nx,Nt,3])
FHLL = np.zeros([Nx-1,Nt,3])
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

## MAIN LOOP
count = 0 # Diagnostics/progress indication
for n in range(Nt-1):
	count += 1
	L = LFunc(U,F,dx,n)
	U[1:-1,n+1,:] = U[1:-1,n,:] + dt*L
	# Update all fluxes based on this update
	F = fluxUpdate(U,F,n)
	# Reinforce BCS
	U[0,n+1,:] = U[0,0,:]
	F[0,n+1,:] = F[0,0,:]
	U[-1,n+1,:] = U[-1,0,:]
	F[-1,n+1,:] = F[-1,0,:]
	if count % 1000 == 0:
		print count*100/Nt , '% Done'
		
## CONVERGENCE CALCULATIONS
xE, rhoE, vE, pE = ee.riemann(xMin, xMax, xMax/2., Nx, tMax, rhoL, vL, pL, rhoR, vR, pR, gamma,TOL=1.0e-14, MAX=100)
#diff = np.abs(rhoE - U[:,-1,0])
L1   = (dx*diff).sum()
print Nx , L1

## PLOTTING
plt.figure()
plt.title('Density in the Sod Tube Problem at t=0.2', fontsize = 24)
plt.xlabel('X Position' , fontsize = 18)
plt.ylabel(r'Density', fontsize =18)
plt.tick_params(labelsize=14)
plt.xlim([xMin,xMax])
plt.plot(xPoints,U[:,-1,0], label = 'Computed Density', lw = 3, color = tableau20[0])
plt.plot(xE     , rhoE    , label = 'Exact Solution', lw = 3, color = tableau20[2])
plt.ylim([0,rhoL*1.1])
plt.legend()
plt.show()

