import numpy as np
import pylab as plt
import time
def minmod(x,y,z):
	# Minmod function
	return 0.25*np.abs(np.sign(x) + np.sign(y))*(np.sign(x) + np.sign(z)) * np.minimum(np.abs(x),np.abs(y),np.abs(z))
	
N = 1000
A = np.arange(N)
Aavg1 = np.zeros(len(A)-1)
Aavg2 = np.zeros(len(A)-1)
# Average adjacent elements by looping
time1 = time.time()
for i in range(N-1):
	Aavg1[i] = A[i]/2. + A[i+1]/2. 

loopTime = time.time() - time1

#Average adjacent elements at once
time2 = time.time()
Aavg2 = A[0:-1]/2. + A[1::]/2.
arrayTime = time.time() - time2

print 'Loop time:',loopTime
print 'Array time:',arrayTime
print 'Factor', loopTime/arrayTime

A = np.arange(5)
B = .2*A
C = 3*A
print minmod(A,B,C)
