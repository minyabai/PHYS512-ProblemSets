## PHYS512 Problem Set 5
## ========================
## Minya Bai (260856843)
## Question 2

import numpy as np
from matplotlib import pyplot as plt
from functions import corr

x = np.linspace(-5,5,2001)
f = np.exp(-x**2)
g = np.exp(-x**2)

plt.plot(corr(f,g))
plt.show()
