import numpy as np


rhos = np.random.rand(4,3)
rhos2 = np.random.rand(3,4)

rhoMax = max(rhos.max(),rhos2.max())

print rhos
print rhos2
print rhoMax
