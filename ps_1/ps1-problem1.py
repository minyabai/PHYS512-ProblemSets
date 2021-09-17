## PHYS512 Problem Set 1
## Minya Bai (260856843)
## ===========================
## Problem 1
## a) From the Taylor series expansion (PDF in ps1 folder), we get the
##    following expression for the estimate of the derivative of f(x):
##    f'(x) ~ (8*f(x+dx)-8*f(x-dx)-f(x+2dx)+f(x-2dx))/(12dx)
## b) dx = (15/2 * eps*f/f^(5))^(1/5)

import numpy as np
from matplotlib import pyplot as plt

## Calculating the derivative
def cal_deriv(x=10,dx=10**-52,a=1): # a - scaling factor
    # defining the four points
    x1=x-2*dx
    x2=x-dx
    x3=x+dx
    x4=x+2*dx

    # calculating the function at the four points
    f1=np.exp(a*x1)
    f2=np.exp(a*x2)
    f3=np.exp(a*x3)
    f4=np.exp(a*x4)

    deriv=(f1-8*f2+8*f3-f4)/(12*dx) # using Taylor series expansion
    # print('The derivative is ',deriv)

    return deriv

## Calculating the optimal delta
def cal_delta(x,a=1):
    eps = 10**-16 # roundoff error
    return (15/2 * np.exp(a*x)/(a**(5)*np.exp(a*x)) * eps)**(1/5)


## ============================
##  Testing the Code
## ============================

print('The optimal delta for f(x) = exp(x) is ~ ',cal_delta(x=10))
print('The optimal delta for f(x) = exp(.01x) is ~ ',cal_delta(x=10,a=0.01))


## Verify the optimal delta by plotting the error as a function of delta
## We should expect the minimum to be close to the optimal delta

delta=np.logspace(-8,1,20)

plt.plot(delta, np.abs(cal_deriv(dx=delta)-np.exp(10)),label='exp(x)')
plt.plot(delta, np.abs(cal_deriv(dx=delta,a=.01)-.01*np.exp(.01*10)),label='exp(.01x)')
plt.yscale('log') # log axes to better see trend in plots
plt.xscale('log')
plt.legend()
plt.show()
