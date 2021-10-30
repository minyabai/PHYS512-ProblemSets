## PHYS512 - Problem Set 5
## ========================
## Minya Bai (260856843)
## Question 3

import numpy as np
from matplotlib import pyplot as plt
from functions import *

a = 1000
x = np.linspace(-5,5,2*a+1)
f = np.exp(-x**2)
f1 = shift_conv(f,a)
f2 = shift_conv(f,500)
f3 = shift_conv(f,800)
f1_corr = corr(f,f1)
f2_corr = corr(f,f2)
f3_corr = corr(f,f3)

# plt.plot(f)
# plt.plot(f1)
plt.plot(f1_corr)
plt.plot(f2_corr)
plt.plot(f3_corr)
plt.show()
