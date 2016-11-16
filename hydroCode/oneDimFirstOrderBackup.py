import numpy as np
import odeSolve
import pylab as plt

def uFunc(r,t):
	r[0] = U # Expect only one entry for r
	LU = -(FPlus-FMinus)/dx
	return LU


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
	
	
def sound(gamma,P,rho):
	# Calculate the speed of sound
	return np.sqrt( gamma*P / rho )
	
## Setting some parameters	
tMin    = 0.
tMax    = 1.
Nt      = 10000
dt      = (tMax-tMin)/Nt

xMin    = 0.
xMax    = 1.
Nx      = 100
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
P     = 1. # Pressure
gamma = 1.4 # Adiabatic index
p     = 1.0 # EOS value
e     = p / ((gamma-1)*rho) # Specific internal energy
E     = rho*e + 0.5 * rho*v**2 # Total energy density

#Left side
U[:len(xPoints)/2,0,0] = rho
U[:len(xPoints)/2,0,1] = rho*v
U[:len(xPoints)/2,0,2] = E
F[:len(xPoints)/2,0,0] = rho*v
F[:len(xPoints)/2,0,1] = rho*v**2 + P
F[:len(xPoints)/2,0,2] = v*(E+P)
# Right side
U[len(xPoints)/2:,0,0] = rho*0.1
U[len(xPoints)/2:,0,1] = rho*0.1*v
U[len(xPoints)/2:,0,2] = E/8.
F[len(xPoints)/2:,0,0] = rho*0.1*v
F[len(xPoints)/2:,0,0] = rho*0.1*v**2 + P
F[len(xPoints)/2:,0,0] = v*(E/8.+P)

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
		soundL = sound(gamma,P,U[i-1,n,0])
		soundC = sound(gamma,P,U[i  ,n,0])
		soundR = sound(gamma,P,U[i+1,n,0])
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
		if count %10000 == 0:
			print count*100/(Nt*Nx), '% Complete'
#plt.plot(xPoints,U[:,0,0], label = 'Initial')
#plt.plot(xPoints,U[:,len(tPoints)/2,0], label = 'Halfway')
#plt.plot(xPoints,U[:,-1,0], label = 'End')
plt.plot(xPoints,U[:,0.2*len(tPoints),0], label = 'Sod Test')
plt.legend()
plt.show()
