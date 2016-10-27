import functions as f
import pylab as plt
import numpy as np

# Values of parameters
#a     = 1.
rho0  = 1.
c0    = 1.
k     = 1.
psi0  = 1.
beta  = np.pi/4.

rhoi  = .656  #rhoi and ci based on Marson 2006 Section 5 
ci    = .719
sigma = ci/c0

nMax  = 20

aPoints = 250
force = np.zeros(aPoints,dtype=complex)
aVec  = np.zeros(aPoints)
count = 0

for a in np.linspace(0.,4.,aPoints): 
	I0 = f.I0(rho0,c0,k,psi0,beta)

	alphas, betas = f.inviscid(rhoi,k,a,sigma,rho0,nMax)

	Yp = f.Yp(k,a,alphas,betas,beta,nMax-1)

	force[count] = f.Fz(a,I0,c0,beta,Yp)
	aVec[count]     = a
	count += 1
	
print(force)
plt.plot(k*aVec,(force))
plt.show()

