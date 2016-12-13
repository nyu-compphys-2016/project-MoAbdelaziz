import numpy as np
import scipy.integrate as si

A = np.array([[1,2,3],[13,5,6]])

print si.simps((si.simps(A)))
