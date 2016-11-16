import numpy as np
import pylab as plt

def flux(alphas,FL,FR,UL,UR):
	# Calculate fluxes 
	alphaPlus   = alphas[0]
	alphaMinus  = alphas[1]
	numerator   = alphaPlus*FL + alphaMinus*FR - alphaPlus*alphaMinus*(UR-UL)
	denominator = alphaPlus + alphaMinus
	return numerator/denominator
	
	
def lambdaF(v,sound):
	# Calculate min and max eigenvalues of the Jacobian of the left/right states
	lambdaPlus  = v + sound
	lambdaMinus = v - sound
	return [ lambdaPlus , lambdaMinus ]


def alpha(lambdas,UL,UR):
	# Calculate alphas from hydrocode handout
	lambdaPlusL  = lambdas[0]
	lambdaMinusL = lambdas[1]
	lambdaPlusR  = lambdas[2]
	lambdaMinusR = lambdas[3]
	
	alphaPlus  = max(0 , lambdaPlusL   ,   lambdaPlusR)
	alphaMinus = max(0 , -lambdaMinusL , -lambdaMinusR)
	return [ alphaPlus , alphaMinus ]
	
	
def sound(gamma,p,rho):
	# Calculate the speed of sound
	return np.sqrt( gamma*p / rho )

def pressure(gamma,rho,e):
	# Calculate pressure from U vector
	return (gamma-1) * rho * e
	
## Setting some parameters	
tMin    = 0.
tMax    = 0.2
Nt      = 2000
dt      = (tMax-tMin)/Nt

xMin    = 0.
xMax    = 1.
Nx      = 200
dx      = (xMax - xMin)/Nx

## Initializing arrays
tPoints = np.linspace(tMin,tMax,Nt)
xPoints = np.linspace(xMin,xMax,Nx)
U = np.zeros([len(xPoints),len(tPoints),3]) 
F = np.zeros([len(xPoints),len(tPoints),3])
lambdas = np.zeros(6)
## Setting the initial conditions
rho   = 1. # Fluid density
v     = 0. # Velocity
P     = 1. # Pressure (need to phase this out, just use p and make sure p updates appropriately)
gamma = 1.4 # Adiabatic index
p     = 1.0 # EOS value (might just be pressure?)
e     = p / ((gamma-1)*rho) # Specific internal energy
E     = rho*e + 0.5 * rho*v**2 # Total energy density

#Left side
U[:len(xPoints)/2,0,0] = rho
U[:len(xPoints)/2,0,1] = rho*v
U[:len(xPoints)/2,0,2] = E
F[:len(xPoints)/2,0,0] = rho*v
F[:len(xPoints)/2,0,1] = rho*v**2 + p
F[:len(xPoints)/2,0,2] = v*(E+P)
# Right side
U[len(xPoints)/2:,0,0] = rho*0.1
U[len(xPoints)/2:,0,1] = rho*0.1*v
U[len(xPoints)/2:,0,2] = rho*e/8. + 0.5 * rho*v**2 *0.1
F[len(xPoints)/2:,0,0] = rho*0.1*v
F[len(xPoints)/2:,0,0] = rho*0.1*v**2 + p/8.
F[len(xPoints)/2:,0,0] = v*(rho*e/8. + 0.5 * rho*v**2 *0.1+ p/8.)

## Main loops
count = 0 # For diagnostics etc
for n in range(Nt-1):
	for i in range(1,Nx-1):
		count += 1
		FL = F[i-1,n,:] # Left
		FC = F[i  ,n,:] # Current
		FR = F[i+1,n,:] # Right
		UL = U[i-1,n,:]
		UC = U[i  ,n,:]
		UR = U[i+1,n,:]
		vL = U[i-1,n,1]/U[i-1,n,0] # This gets just the velocity!
		vC = U[i  ,n,1]/U[i  ,n,0] 
		vR = U[i+1,n,1]/U[i+1,n,0]
		
		
		soundL = sound(gamma,(gamma-1)*(U[i-1,n,2] - 0.5*U[i-1,n,1]*U[i-1,n,1]/U[i-1,n,0]),U[i-1,n,0])
		soundC = sound(gamma,(gamma-1)*(U[i  ,n,2] - 0.5*U[i  ,n,1]*U[i  ,n,1]/U[i  ,n,0]),U[i  ,n,0])
		soundR = sound(gamma,(gamma-1)*(U[i+1,n,2] - 0.5*U[i+1,n,1]*U[i+1,n,1]/U[i+1,n,0]),U[i+1,n,0])
		lambdasL = lambdaF(vL,soundL)
		lambdasR = lambdaF(vR,soundR)
		lambdasC = lambdaF(vC,soundC)
		lambdas[0:2] = lambdasL
		lambdas[2:4] = lambdasC
		lambdas[4:6] = lambdasR
		
		alphasiMinus  = alpha(lambdas[0:4],UL,UC)
		FiMinus       = flux(alphasiMinus,FL,FC,UL,UC)
		alphasiPlus   = alpha(lambdas[2:6],UC,UR)
		FiPlus        = flux(alphasiPlus,FC,FR,UC,UR)
		L = -(FiPlus - FiMinus)/dx
		U[i,n+1,:] = U[i,n,:] + dt*L
		F[i,n+1,0] = U[i,n+1,1]
		F[i,n+1,1] = (gamma-1)*(U[i,n+1,2] - 0.5*U[i,n+1,1]*U[i,n+1,1]/U[i,n+1,0]) + U[i,n+1,1]*U[i,n+1,1]/U[i,n+1,0]
		F[i,n+1,2] = ((gamma-1)*(U[i,n+1,2] - 0.5*U[i,n+1,1]*U[i,n+1,1]/U[i,n+1,0])  + U[i,n+1,2]) * U[i,n+1,1]/U[i,n+1,0]
		
		# Fix ghost cells (first and last points)
		U[0,n+1,:] = U[0,0,:]
		F[0,n+1,:] = F[0,0,:]
		U[-1,n+1,:] = U[-1,0,:]
		F[-1,n+1,:] = F[-1,0,:]
		if count %10000 == 0:
			print count*100/(Nt*Nx), '% Complete'

plt.figure()
plt.title("Sod Shock Tube Density Plot" , fontsize = 24)
plt.xlabel("Position" , fontsize = 18)
plt.ylabel("Mass Density" , fontsize = 18)
plt.plot(xPoints,U[:,0,0], label = 'Initial')
plt.plot(xPoints,U[:,Nt/25,0], label = '25%')
plt.plot(xPoints,U[:,Nt/2,0], label = 'Halfway')
plt.plot(xPoints,U[:,-1,0], label = 'Sod Test')
plt.ylim([0,1.05])
plt.legend()
plt.show()
