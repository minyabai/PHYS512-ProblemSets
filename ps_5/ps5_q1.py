## PHYS512 - Problem Set 5
## ============================
## Minya Bai (260856843)
## Question 1

import numpy as np
from matplotlib import pyplot as plt
from functions import shift_conv

a = 1000 ## shift value
x = np.linspace(-5,5,a*2+1)
f = np.sin(x) # np.exp(-x**2)

h = shift_conv(f,a)

plt.plot(f)
plt.plot(h)
plt.show()
