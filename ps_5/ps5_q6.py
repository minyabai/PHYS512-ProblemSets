## PHYS512 - Problem Set 5
## ===========================
## Minya Bai (260856843)
## Question 6

import numpy as np
from matplotlib import pyplot as plt
from functions import corr

n = 500
randwalk = np.cumsum(np.random.randn(n)) # gen rand walk

plt.plot(randwalk)
plt.show()

N = np.arange(n, dtype='float')[1:]**-2 # we expect to scale by k**-2
f2 = corr(randwalk,randwalk) # calculate correlation function
fft_f2 = np.fft.fft(f2) # ft to get power spectrum


plt.plot(np.fft.fft(f2))
plt.plot(np.max(fft_f2)*N)
plt.show()


