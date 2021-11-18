## PHYS512 - Problem Set 7
## ===========================
## Minya Bai (260856843)
## Question 3

import numpy as np
from matplotlib import pyplot as plt

n = 1000000
u = np.random.rand(n)
v = np.random.rand(n)
p = 1.0
ratio = p * v/u

# We want the region to be bounded by 0 \leq u \leq [p(v/u)]**0.5
# where p(x) = exp(-x)
cut = u < np.exp(-ratio) ** 0.5
deviates = ratio[cut]
expdev = np.random.exponential(p,len(deviates))

print('Accepted fraction: ', len(deviates)/len(u))

deviates.sort()
expdev.sort()

## Plot of histogram
plt.hist(deviates, bins=200, density=True)
plt.hist(expdev, bins = 200, density=True)
plt.show()
