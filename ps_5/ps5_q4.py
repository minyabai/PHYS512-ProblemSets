## PHYS512 - Problem Set 5
## ===========================
## Minya Bai (260856843)
## Question 4

import numpy as np
from matplotlib import pyplot as plt

def conv_safe(f,g):
    d = np.abs(len(f)-len(g)) # find diff in length to make sure initially same size
    n = max(len(f),len(g)) # find max of length between two
    
    if len(f) > len(g): # match shorter to longer with 0s
        g = np.concatenate((g,np.zeros(d)))
    if len(g) > len(f):
        f = np.concatenate((f,np.zeros(d)))

    f = np.concatenate((f,np.zeros(n))) # padded the ends with 0s for both
    g = np.concatenate((g,np.zeros(n)))
    h = np.fft.irfft(np.fft.fft(f)*np.fft.fft(g),len(f)) # convolution

    return h

x = np.linspace(-5,5,2001)
f = np.exp(-x**2)
g = np.exp(-(x-2)**2)

h = conv_safe(f,g)
plt.plot(f, label='gaussian 1')
plt.plot(g, label='gaussian 2')
plt.plot(h/max(h), label='scaled convolution')
plt.legend()
plt.show()

