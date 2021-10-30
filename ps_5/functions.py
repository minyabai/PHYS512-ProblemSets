# functions for q1 and q2 of ps5

import numpy as np

def shift_conv(x,a):
    k = np.arange(len(x))
    g = np.exp(-2*np.pi*1J*a*k/len(x)) # how much we wnat to shift it
    h = np.fft.irfft(np.fft.fft(x)*g,len(x)) # convolution and inverse ft

    return h

def corr(f,g):
    return np.fft.ifft(np.fft.fft(f)*np.fft.fft(np.flip(g)))
    # return np.fft.ifft(np.fft.fft(f)*np.conj(np.fft.fft(g)))
