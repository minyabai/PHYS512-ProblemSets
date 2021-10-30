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

# plt.plot(x,dft)
plt.plot(x,fft)
# plt.yscale('log')
# plt.ylim(1e-16,1e-12)
# plt.plot(x,fft-dft)
# plt.show()

## Part d
## ====================
window = 0.5 - 0.5 * np.cos(2*np.pi*x/N)
fft = np.abs(np.fft.fft(np.sin(2*np.pi*k*x/N)*window))

plt.plot(x,fft)
# plt.show()

## Part e
## ====================
fft_window = np.real(np.fft.fft(window))

# plt.plot(x,fft_window)
# plt.show()

# sum3 = (1-np.exp(-2*np.pi*1J*(x-k-1)))/(1-np.exp(-2*np.pi*1J*(x-k-1)/N))
# sum4 = (1-np.exp(-2*np.pi*1J*(x+k-1)))/(1-np.exp(-2*np.pi*1J*(x+k-1)/N))
# sum5 = (1-np.exp(-2*np.pi*1J*(x-k+1)))/(1-np.exp(-2*np.pi*1J*(x-k+1)/N))
# sum6 = (1-np.exp(-2*np.pi*1J*(x+k+1)))/(1-np.exp(-2*np.pi*1J*(x+k+1)/N))

# sum3 = np.sin(2*np.pi*k*(x-1)/N)/4 + np.sin(2*np.pi*k*x/N)/2 + np.sin(2*np.pi*k*(x+1)/N)/4
func = np.fft.fft(np.sin(2*np.pi*k*x/N))
fft_w = -np.roll(func,1)/4 + func/2 - np.roll(func,-1)/4
# fft_w = np.fft.rfft(fft_w)

# dft = np.abs(np.real(1/(8J)*(2*(sum1-sum2)-sum3+sum4-sum5+sum6)))
# dft = np.abs(np.real(np.fft.fft(sum3)))
# fft_sin = np.fft.fft(np.sin(2*np.pi*k*x/N))
plt.plot(x,np.abs(fft_w))
# plt.plot(x,dft)
plt.show()
