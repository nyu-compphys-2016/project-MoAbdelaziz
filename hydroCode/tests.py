import numpy as np

A = np.zeros([7,7])
A[0:3,:] = 1.
A[3::,:] = 0.5
A[2:5,2:5] = np.random.rand(3,3)

print 'array'
print A
print 'avg btwn rows'
print (A[1:,:] + A[:-1,:])/2.
print 'avg btwn cols'
print (A[:,1:]+A[:,:-1])/2.

print 'summed over overlapping region'
print (A[1:,:] + A[:-1,:])/2. + (A[:,1:]+A[:,:-1])/2.
