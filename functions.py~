import numpy as np
import pylab as plt
import scipy.special as ss

def Fz(a,I0,c0,beta,Yp):
	# Compute the axial radiation force on a sphere due to a Bessel beam
	# a    : Sphere radius
	# I0   : Function of other parameters
	# c0   : Phase velocity of the surrounding fluid
	# beta : Cone angle of the planar wave components, relative to the z-axis
	# Yp   : Function of other parameters
	
	return (np.pi * a**2) * (I0/c0) * (1./np.cos(beta)) * Yp

def I0(rho0,c0,k,psi0,beta):
	# Compute Acoustic Intensity (W/m^2) for use in Fz
	# rho0 : Fluid density (surrounding) 
	# c0   : Phase velocity of the surrounding fluid
	# k    : Wavenumber
	# psi0 : Beam amplitude
	# beta : Cone angle of the planar wave components, relative to the z-axis
	
	return (rho0 * c0)/2. * (k * psi0)**2 * np.cos(beta)
	
def Yp(k,a,alphas,betas,beta,nMax):
	# Compute the function Yp for use in Fz
	# k      : Wavenumber
	# a      : Sphere radius
	# alphas : (Re(sn)-1)/2 , for partial wave coefficients sn
	# betas  : (Im(sn))/2
	# beta   : Cone angle of the planar wave components, relative to the z-axis
	# nMax   : max n value to sum to (ideal = infinity)
	
	Yp = 0j #initialize
	for n in range(nMax):
		Yp += (n+1) * \
			  (alphas[n] + alphas[n+1] + 2*(alphas[n]*alphas[n+1] + betas[n]*betas[n+1]))  * \
			  ss.legendre(n)(np.cos(beta)) * ss.legendre(n+1)(np.cos(beta))
	return -(2/(k*a))**2 * Yp

def inviscid(rhoi,k,a,sigma,rho0,nMax):
	# Compute alphas and betas for an inviscid fluid sphere
	# rhoi  : Density of the sphere
	# k     : Wavenumber
	# a     : Sphere radius
	# sigma : Ratio of sound velocity in sphere to sound velocity in surrounding material
	# rho0  : Density of surrounding fluid
	D = np.zeros(nMax,dtype=complex)
	for n in range(nMax):
		D[n] = rhoi*k*a*ss.spherical_jn(n,(k*a/sigma)) * \
		(ss.spherical_jn(n,k*a,derivative = True) + 1.j*ss.spherical_yn(n,k*a,derivative=True))- \
		rho0*k*a/sigma * ss.spherical_jn(n,(k*a/sigma),derivative=True) * \
		(ss.spherical_jn(n,k*a) + 1.j*ss.spherical_yn(n,k*a)) 
		
	s = -np.conj(D)/D 
	alphas = (np.real(s)-1.)/2.
	betas  =  np.imag(s)/2.
	return  alphas,betas
