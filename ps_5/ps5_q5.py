## PHYS512 - Problem Set 5
## ==========================
## Minya Bai (260856843)
## Question 5

import numpy as np
from matplotlib import pyplot as plt

## Part c
## ====================
N = 100
x = np.arange(N)
k = 51/2

sum1 = (1-np.exp(-2*np.pi*1J*(x-k)))/(1-np.exp(-2*np.pi*1J*(x-k)/N))
sum2 = (1-np.exp(-2*np.pi*1J*(x+k)))/(1-np.exp(-2*np.pi*1J*(x+k)/N))

dft = np.abs(1/(2J)*(sum1-sum2))
fft = np.abs(np.fft.fft(np.sin(2*np.pi*k*x/N)))

# print(np.mean(fft-dft))

# plt.plot(x,dft,label='dft')
# plt.plot(x,fft,label='without window function')
# plt.yscale('log')
# plt.ylim(1e-16,1e-12)
# plt.plot(x,fft-dft)
# plt.legend()
# plt.show()

## Part d
## ====================
window = 0.5 - 0.5 * np.cos(2*np.pi*x/N)
fft = np.abs(np.fft.fft(np.sin(2*np.pi*k*x/N)*window)) # ft with window function

plt.plot(x,fft,label='window 1')
# plt.legend()
# plt.show()

## Part e
## ====================
fft_window = np.real(np.fft.fft(window)) # fft of window function

# plt.plot(x,fft_window)
# plt.show()

func = np.fft.fft(np.sin(2*np.pi*k*x/N)) # ft of the initial function
fft_w = -np.roll(func,1)/4 + func/2 - np.roll(func,-1)/4 # lin comb with neighbour terms

plt.plot(x,np.abs(fft_w),label='window 2')
plt.legend()
plt.show()
