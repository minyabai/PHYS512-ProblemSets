## PHYS 512 - Problem Set 2
## Minya Bai (260856843)
## ===========================
## Question 3

import numpy as np
from matplotlib import pyplot as plt

def log2_fit():
    # rescale (0.5,1)  to be in region of (-1,1)
    x = np.linspace(0.5,1,100)
    X = np.interp(x,(0.5,1),(-1,1))

    # get the chebyshev fit coefficients
    coeffs = np.polynomial.chebyshev.chebfit(X,np.log2(x),30)

    # cutting the data to only take coeffs better than 10^-6 in abs magnitude
    cut = np.abs(coeffs) >= 10**-5
    coeffs = coeffs[cut]

    return coeffs

def mylog2(x):
    # we know from log laws that log_a(b) = log_d(b)/log_d(a), we can use this
    # to evaluate ln using log2

    # get the chebyshev coeff for fit
    log2_cfit = log2_fit()

    # get the mantissa and the exp
    mant_x, exp_x = np.frexp(x)
    mant_e, exp_e = np.frexp(np.e)

    # rescale the mantissa to the region we want
    mant_X, mant_E = np.interp([mant_x,mant_e],(0.5,1),(-1,1))

    # calculate log_2(x) and log_2(e)
    log2_x = np.polynomial.chebyshev.chebval(mant_X, log2_cfit) + exp_x
    log2_e = np.polynomial.chebyshev.chebval(mant_E, log2_cfit) + exp_e

    return log2_x/log2_e
    
## ===========================
## Testing Code
## ===========================
## Part a)
print("We need to keep {} terms.".format(len(log2_fit())))

## Part b)
x = np.linspace(0.5,10,100)
log2x = [mylog2(i) for i in x]

error = log2x - np.log(x)

print(error)

plt.plot(x,np.log(x),label = 'np.log')
plt.plot(x,log2x,'.',label = 'mylog2')
plt.legend()
plt.show()

