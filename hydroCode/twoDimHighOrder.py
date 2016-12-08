import numpy as np
import pylab as plt
import eulerExact as ee
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
import initialConds

showAnim = 1 
saveAnim = 1
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

def LFunc(U,F,dx,dy,n):
	# Calculate L(U) given a U[:,:,n,0:2] for the conserved variable array at time n, all positions
	FHLLH ,FHLLV= FHLLFunc(U,F,n) 

	#print FHLL[:,n,0]
	L = -(1/dx) * (FHLLH[1::,2:-2,n,:] - FHLLH[0:-1,2:-2,n,:]) - (1/dy) * (FHLLV[2:-2,1::,n,:] - FHLLV[2:-2,0:-1,n,:])
	return L
	
def FHLLFunc(U,F,n):
	# Calculate HLL Flux as an array FHLLH[:,:,0:2] and FHLLV at all interfaces (3 less entries than cell centers for 2nd order space)
	
	rho = U[:,:,n,0]
	vx = U[:,:,n,1]/U[:,:,n,0]
	vy = U[:,:,n,2]/U[:,:,n,0]
	p = F[:,:,n,1] - U[:,:,n,1]**2 / U[:,:,n,0]
	lambdaPlusL , lambdaMinusL ,lambdaPlusR ,lambdaMinusR ,lambdaPlusD, lambdaMinusD, lambdaPlusU, lambdaMinusU = lambdaFunc(vx,vy,gamma,p,rho)
	alphaPlusH ,alphaMinusH, alphaPlusV, alphaMinusV = alphaFunc(lambdaPlusL,lambdaMinusL,lambdaPlusR,lambdaMinusR,lambdaPlusD,lambdaMinusD,lambdaPlusU,lambdaMinusU)

	lefts, rights, downs, ups = interpolate(vx,vy,gamma,p,rho)
	FHLLH = np.zeros([Nx-3,Ny,Nt,4])

	UL = np.zeros([Nx-3,Ny,4])
	UR = np.zeros([Nx-3,Ny,4])
	FL = np.zeros([Nx-3,Ny,4])
	FR = np.zeros([Nx-3,Ny,4])
	EL = lefts[:,:,0]/(gamma-1) + 0.5* lefts[:,:,1]*(lefts[:,:,2]**2+lefts[:,:,3]**2)  # energy (used multiple times)
	ER = rights[:,:,0]/(gamma-1) + 0.5* rights[:,:,1]*(rights[:,:,2]**2 +rights[:,:,3]**2)
	UL[:,:,0], UR[:,:,0] = lefts[:,:,1], rights[:,:,1]
	UL[:,:,1], UR[:,:,1] = lefts[:,:,1]*lefts[:,:,2] , rights[:,:,1]*rights[:,:,2]
	UL[:,:,2], UR[:,:,2] = lefts[:,:,1]*lefts[:,:,3] , rights[:,:,1]*rights[:,:,3]
	UL[:,:,3], UR[:,:,3] = EL , ER
	FL[:,:,0], FR[:,:,0] = lefts[:,:,1]*lefts[:,:,2] , rights[:,:,1]*rights[:,:,2]
	FL[:,:,1], FR[:,:,1] = lefts[:,:,1]*lefts[:,:,2]**2 + lefts[:,:,0] , rights[:,:,1]*rights[:,:,2]**2 + rights[:,:,0] 
	FL[:,:,2], FR[:,:,2] =  lefts[:,:,1]*lefts[:,:,2]*lefts[:,:,3] , rights[:,:,1]*rights[:,:,2]*rights[:,:,3]
	FL[:,:,3], FR[:,:,3] = (EL + lefts[:,:,0])*lefts[:,:,2] , (ER + rights[:,:,0])*rights[:,:,2]

	FHLLH[:,:,n,:] = (alphaPlusH[:,:,None]*FL[:,:,:] + alphaMinusH[:,:,None]*FR[:,:,:] - alphaPlusH[:,:,None]*alphaMinusH[:,:,None] *(UR[:,:,:]- UL[:,:,:]))/(alphaPlusH[:,:,None] + alphaMinusH[:,:,None])

	FHLLV = np.zeros([Nx,Ny-3,Nt,4])
	UD = np.zeros([Nx,Ny-3,4])
	UU = np.zeros([Nx,Ny-3,4])
	GD = np.zeros([Nx,Ny-3,4])
	GU = np.zeros([Nx,Ny-3,4])
	ED = downs[:,:,0]/(gamma-1) + 0.5* downs[:,:,1]*(downs[:,:,2]**2+downs[:,:,3]**2) # energy (used multiple times)
	EU = downs[:,:,0]/(gamma-1) + 0.5* ups[:,:,1]*(ups[:,:,2]**2 +ups[:,:,3]**2)
	UD[:,:,0], UU[:,:,0] = downs[:,:,1], ups[:,:,1]
	UD[:,:,1], UU[:,:,1] = downs[:,:,1]*downs[:,:,2] , ups[:,:,1]*ups[:,:,2]
	UD[:,:,2], UU[:,:,2] = downs[:,:,1]*downs[:,:,3] , ups[:,:,1]*ups[:,:,3]
	UD[:,:,3], UU[:,:,3] = ED , EU
	GD[:,:,0], GU[:,:,0] = downs[:,:,1]*downs[:,:,3] , ups[:,:,1]*ups[:,:,3]
	GD[:,:,1], GU[:,:,1] = downs[:,:,1]*downs[:,:,2]*downs[:,:,3], ups[:,:,1]*ups[:,:,2]*ups[:,:,3]
	GD[:,:,2], GU[:,:,2] = downs[:,:,1]*downs[:,:,3]**2 + downs[:,:,0] , ups[:,:,1]*ups[:,:,3]**2 + ups[:,:,0]
	GD[:,:,3], GU[:,:,3] = (ED + downs[:,:,0])*downs[:,:,3] , (EU + ups[:,:,0])*ups[:,:,3]
	
	FHLLV[:,:,n,:] = (alphaPlusV[:,:,None]*GD[:,:,:] + alphaMinusV[:,:,None]*GU[:,:,:] - alphaPlusV[:,:,None]*alphaMinusV[:,:,None] *(UU[:,:,:]- UD[:,:,:]))/(alphaPlusV[:,:,None] + alphaMinusV[:,:,None])
	
	return FHLLH, FHLLV
	
def lambdaFunc(vx,vy,gamma,p,rho):
	# Calculate lambda as an array at all positions lambdas[:,:] at some time 
	# v[:,:] at all positions
	# gamma constant
	# p[:,:] at all positions	
	# rho[:,:] at all positions

	lefts, rights, downs, ups  = interpolate(vx,vy,gamma,p,rho)
	
	soundL = np.sqrt(gamma*lefts[:,:,0]/lefts[:,:,1])
	soundR = np.sqrt(gamma*rights[:,:,0]/rights[:,:,1])
	lambdaPlusL  = lefts[:,:,2] + soundL
	lambdaMinusL = lefts[:,:,2] - soundL
	lambdaPlusR  = rights[:,:,2] + soundR
	lambdaMinusR = rights[:,:,2] - soundR
	
	soundD = np.sqrt(gamma*downs[:,:,0]/downs[:,:,1])
	soundU = np.sqrt(gamma*ups[:,:,0]/ups[:,:,1])
	lambdaPlusD  = downs[:,:,2] + soundD
	lambdaMinusD = downs[:,:,2] - soundD
	lambdaPlusU  = ups[:,:,2] + soundU
	lambdaMinusU = ups[:,:,2] - soundU
	
	return lambdaPlusL, lambdaMinusL, lambdaPlusR, lambdaMinusR, lambdaPlusD, lambdaMinusD, lambdaPlusU, lambdaMinusU
	
def alphaFunc(lambdaPlusL,lambdaMinusL,lambdaPlusR,lambdaMinusR,lambdaPlusD,lambdaMinusD,lambdaPlusU,lambdaMinusU):
	# Find alphas at all interfaces (3 less entries than cell positions for 2nd order space)
	dimsH = lambdaPlusL.shape
	dimsV = lambdaPlusD.shape
	alphaPlusH  = np.maximum.reduce([np.zeros(dimsH),lambdaPlusL,lambdaPlusR])
	alphaMinusH = np.maximum.reduce([np.zeros(dimsH),-lambdaMinusL,-lambdaMinusR])
	# H: Horizontal, V: Vertical
	alphaPlusV  = np.maximum.reduce([np.zeros(dimsV),lambdaPlusD,lambdaPlusU])
	alphaMinusV = np.maximum.reduce([np.zeros(dimsV),-lambdaMinusD,-lambdaMinusU]) 
	#print alphaPlusV
	return alphaPlusH ,alphaMinusH, alphaPlusV, alphaMinusV

def energyFunc(p,gamma,rho,vx,vy):
	# Calculate energy in terms of p, gamma, rho, v
	return p/(gamma-1) + 0.5* rho*(vx*vx+vy*vy)

def fluxUpdate(U,F,n):
	# Calculate updated values of F and G based on a new U
	newRho  = U[:,:,n+1,0]
	newRhoVx = U[:,:,n+1,1]
	newRhoVy = U[:,:,n+1,2]
	newVx    = newRhoVx/newRho
	newVy   = newRhoVy/newRho
	newE    = U[:,:,n+1,3]
	newP    = (gamma-1) * (newE - 0.5*newRho*(newVx**2+newVy**2))
	
	F[:,:,n+1,0] = newRho*newVx
	F[:,:,n+1,1] = newRho*newVx**2 + newP
	F[:,:,n+1,2] = newRho*newVx*newVy
	F[:,:,n+1,3] = (newE+newP)*newVx
	
	return F
	
def minmod(x,y,z):
	# Minmod function
	return 0.25*np.fabs(np.sign(x) + np.sign(y))*(np.sign(x) + np.sign(z)) * np.minimum.reduce([np.fabs(x),np.fabs(y),np.fabs(z)])
	
def interpolate(vx,vy,gamma,p,rho):
	#Interpolate states at interface sides for second order
	theta = 1.5
	cs = np.zeros([Nx,Ny,4]) # Cell States (values at cell centers)
	cs[:,:,0] = p
	cs[:,:,1] = rho
	cs[:,:,2] = vx 
	cs[:,:,3] = vy
	
	lefts  = cs[1:-2,:,:] + 0.5* minmod(theta*(cs[1:-2,:,:] - cs[0:-3,:,:]) , 0.5*(cs[2:-1,:,:] - cs[0:-3,:,:]) , theta*(cs[2:-1,:,:] - cs[1:-2,:,:]))
	rights = cs[2:-1,:,:] - 0.5* minmod(theta*(cs[2:-1,:,:] - cs[1:-2,:,:]) , 0.5*(cs[3::,:,:] - cs[1:-2,:,:]) , theta*(cs[3::,:,:] - cs[2:-1,:,:]))
	
	
	downs  = cs[:,1:-2,:] + 0.5* minmod(theta*(cs[:,1:-2,:] - cs[:,0:-3,:]) , 0.5*(cs[:,2:-1,:] - cs[:,0:-3,:]) , theta*(cs[:,2:-1,:] - cs[:,1:-2,:]))
	ups    = cs[:,2:-1,:] - 0.5* minmod(theta*(cs[:,2:-1,:] - cs[:,1:-2,:]) , 0.5*(cs[:,3::,:] - cs[:,1:-2,:]) , theta*(cs[:,3::,:] - cs[:,2:-1,:]))
	#print downs[:,:,1]
	return lefts, rights, downs, ups

## COMPUTATION PARAMETERS
tMin    = 0.
tMax    = 1.0
Nt      = 500
dt      = (tMax-tMin)/Nt
tPoints = np.linspace(tMin,tMax,Nt)

#CFL = 1.0

xMin    = 0.
xMax    =  1.
Nx      = 64
dx      = (xMax - xMin)/Nx

yMin    = 0.
yMax    = 1.
Ny      = 64
dy      = (yMax - yMin)/Ny


xPoints = np.linspace(xMin,xMax,Nx)
yPoints = np.linspace(yMin,yMax,Ny)

## INITIAL CONDITIONS
gamma = 1.4 # Adiabatic index
p0    = 1.0
rho0  = 1.0 
v0    = 0.0

rhoL = 1.0
pL   = 1.0
vxL   = 0.0
vyL =0.0
pR   = 0.125
rhoR = 0.1
vxR   = 0.0
vyR = 0.0
U = np.zeros([Nx,Ny,Nt,4])
F = np.zeros([Nx,Ny,Nt,4])

FHLL = np.zeros([Nx-3,Ny-3,Nt,4]) #N-3 interfaces for N cells with 2 ghost cells on each end
U1 = np.zeros(U.shape)
U2 = np.zeros(U.shape)

#U, F = initialConds.sod1D(U,F,gamma, rhoL,pL,vxL,vyL, rhoR,pR,vxR,vyR)
#U, F = initialConds.bessel(U,F,xPoints,gamma ,1.0,1.0,1.0)
#U,F =   initialConds.kelvHelm(U,F,xPoints,yPoints,gamma,2.5,1.0,2.0,-0.5,0.5,0.035,10**(-1))
U,F = initialConds.kelvHelm2(U,F,xPoints,yPoints,gamma,1.0,1.0,10**(-2),10**(-1),4)

U1 = U.copy() # Need to maintain BCS for U1 and U2 as well
U2 = U.copy()
F1 = F.copy()
F2 = F.copy()


## MAIN LOOP
count = 0 # Track loop number
for n in range(Nt-1):
	count +=1 

	#RK3 Updating
	
	#U[2:-2,2:-2,n+1,:]  = U[2:-2,2:-2,n,:] + dt*LFunc(U,F,dx,dy,n) # FIRST ORDER TIME TO TEST OTHER STUFF
	
	U1[2:-2,2:-2,n,:]  = U[2:-2,2:-2,n,:] + dt*LFunc(U,F,dx,dy,n)

	F1 = fluxUpdate(U1,F,n-1)

	U2[2:-2,2:-2,n,:]  = (3./4.)*U[2:-2,2:-2,n,:] + (1./4.)*U1[2:-2,2:-2,n,:] + (1./4.)*dt*LFunc(U1,F1,dx,dy,n)
	F2= fluxUpdate(U2,F1,n-1)

	U[2:-2,2:-2,n+1,:] = (1./3.)*U[2:-2,2:-2,n,:] + (2./3.)*U2[2:-2,2:-2,n,:] + (2./3.)*dt*LFunc(U2,F2,dx,dy,n)
	# Update all fluxes based on this update
	F = fluxUpdate(U,F,n)
	#print U[:,:,n,0]
##	# Reinforce BCS (return ghost cells to initial conditions) (Need two ghost cells on each side for 2nd order spatial)
#	U[0:2,:,n+1,:]  = U[0:2,:,0,:]
#	F[0:2,:,n+1,:]  = F[0:2,:,0,:]

#	U[-2::,:,n+1,:] = U[-2::,:,0,:]
#	F[-2::,:,n+1,:] = F[-2::,:,0,:]

#	
#	U[:,0:2,n+1,:]  = U[:,0:2,0,:]
#	F[:,0:2,n+1,:]  = F[:,0:2,0,:]

#	U[:,-2::,n+1,:] = U[:,-2::,0,:]
#	F[:,-2::,n+1,:] = F[:,-2::,0,:]

#	# Try Outflow BCS
#	U[0,:,n+1,:]  = U[2,:,n+1,:]
#	F[0,:,n+1,:]  = F[2,:,n+1,:]
#	U[-1,:,n+1,:] = U[-3,:,n+1,:]
#	F[-1,:,n+1,:] = F[-3,:,n+1,:]
#	
#	U[:,0,n+1,:]  = U[:,2,n+1,:]
#	F[:,0,n+1,:]  = F[:,2,n+1,:]
#	U[:,-1,n+1,:] = U[:,-3,n+1,:]
#	F[:,-1,n+1,:] = F[:,-3,n+1,:]
#	
#	U[1,:,n+1,:]  = U[2,:,n+1,:]
#	F[1,:,n+1,:]  = F[2,:,n+1,:]
#	U[-2,:,n+1,:] = U[-3,:,n+1,:]
#	F[-2,:,n+1,:] = F[-3,:,n+1,:]
#	
#	U[:,1,n+1,:]  = U[:,2,n+1,:]
#	F[:,1,n+1,:]  = F[:,2,n+1,:]
#	U[:,-2,n+1,:] = U[:,-3,n+1,:]
#	F[:,-2,n+1,:] = F[:,-3,n+1,:]

	# Try fixed BCS in x, periodic in y
	U[0,:,n+1,:]  = U[0,:,0,:]
	F[0,:,n+1,:]  = F[0,:,0,:]
	U[-1,:,n+1,:] = U[-1,:,0,:]
	F[-1,:,n+1,:] = F[-1,:,0,:]
	
	U[:,0,n+1,:]  = (U[:,-3,n+1,:]+U[:,2,n+1,:])/2.
	F[:,0,n+1,:]  = (F[:,-3,n+1,:]+F[:,2,n+1,:])/2.
	U[:,-1,n+1,:] = (U[:,-3,n+1,:]+U[:,2,n+1,:])/2.
	F[:,-1,n+1,:] = (F[:,-3,n+1,:]+F[:,2,n+1,:])/2.
	
	U[1,:,n+1,:]  = U[1,:,0,:]
	F[1,:,n+1,:]  = F[1,:,0,:]
	U[-2,:,n+1,:] = U[-2,:,0,:]
	F[-2,:,n+1,:] = F[-2,:,0,:]
	
	U[:,1,n+1,:]  =(U[:,-3,n+1,:]+U[:,2,n+1,:])/2.
	F[:,1,n+1,:]  = (F[:,-3,n+1,:]+F[:,2,n+1,:])/2.
	U[:,-2,n+1,:] = (U[:,-3,n+1,:]+U[:,2,n+1,:])/2.
	F[:,-2,n+1,:] =(F[:,-3,n+1,:]+F[:,2,n+1,:])/2.
	F= fluxUpdate(U,F,n)
	U1 = U.copy()
	U2 = U.copy()
	F1 = F.copy()
	F2 = F.copy()

	if count % 50 == 0:
		print count*100/Nt , '% Done'
		
#print F[:,:,0,2]
#print F[:,:,-1,2]

## PLOTTING
fig = plt.figure()
ax = fig.gca()

X, Y = np.meshgrid(xPoints, yPoints)


surf = plt.contourf(X[:,:],Y[:,:],U[:,:,-1,0]) #plots 
plt.colorbar(surf)

#print 'Final Density'
#print U[:,:,-1,0]
#plt.title('Final Density Profile', fontsize = 24)
plt.xlabel('X Position' , fontsize = 18)
plt.ylabel('Y Position', fontsize =18)
plt.tick_params(labelsize=14)
plt.ylim([yMin ,yMax])
plt.xlim([xMin ,xMax])
#ax.set_zlim([rhoR, rhoL])
#plt.legend(loc = 'center')
#ax.view_init(elev=10., azim=60)


def animate(i):
	ax.clear() # Seems necessary to prevent data overlapping between frames
	plt.imshow(U[:,:,i,0] , origin = 'lower')   # update the data
	plt.xlabel('X Position' , fontsize = 18)
	plt.ylabel('Y Position', fontsize =18)
	#ax.set_zlim([rhoR, rhoL])
	return surf,


## Init only required for blitting to give a clean slate.
def init():
	ax.plot_surface([],[],[], rstride=5, cstride=5,cmap=cm.coolwarm)  
	return surf,

ani = animation.FuncAnimation(fig, animate, np.arange(1,Nt,3),  interval=25, blit=False)


ani.save('my_animation.mp4')                           
plt.show()



