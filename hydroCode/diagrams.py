import numpy as np
import pylab as plt
import scipy.special as ss
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

xPoints = np.linspace(-3,3,100)
yPoints = np.linspace(-3,3,100)

rPoints = xPoints[None,:]**2 + yPoints[:,None]**2

f1 = plt.figure()
ax1 = f1.gca()

bessel = np.abs((ss.j0(1.*rPoints)))
plt.imshow(bessel, cmap = cm.seismic)


f2 = plt.figure()
ax2 = f2.gca(projection='3d')

X, Y = np.meshgrid(xPoints,yPoints)

ax2.plot_surface(X,Y,bessel,cmap=cm.Reds,rstride = 1, cstride = 1, linewidth = 0, antialiased=False)  
plt.show(f2)
