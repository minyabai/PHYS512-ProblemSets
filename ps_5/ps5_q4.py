## PHYS512 - Problem Set 5
## ===========================
## Minya Bai (260856843)
## Question 4

import numpy as np
from matplotlib import pyplot as plt

def conv_safe(f,g):
    d = np.abs(len(f)-len(g))
    n = min(len(f),len(g))//2
    
    if len(f) > len(g):
        g = np.concatenate((g,np.zeros(d)))
    if len(g) > len(f):
        f = np.concatenate((f,np.zeros(d)))

    f = np.concatenate((f,np.zeros(n)))
    g = np.concatenate((g,np.zeros(n)))
    h = np.fft.irfft(np.fft.fft(f)*np.fft.fft(g),len(f))

    return h

x = np.linspace(-5,5,2001)
f = np.exp(-x**2)
g = np.exp(-(x+2)**2)

h = conv_safe(f,g)
plt.plot(h)
plt.show()

