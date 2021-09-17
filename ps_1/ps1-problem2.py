## PHYS512 Problem Set 1
## Minya Bai (260856843)
## ===========================
## Problem 2

import numpy as np

def ndiff(fun,x,full=False):
    eps=10**-16
    dx=(3*eps)**(1/3) # approx. f/f^(3) ~ 1
    print(dx)
    deriv = derive(fun,x,dx)
    
    if full:
        # returns derivative, dx, and estimate of error of derivative
        # return deriv,dx,err

        err = 3**(4/3)/2 * eps**(2/3) * deriv
        
        return deriv, dx, err
        
    else:
        # returns numerival derivative
        # return deriv
        return deriv

def derive(fun,x,dx):
    f1=fun(x+dx)
    f2=fun(x-dx)
    print(f1,f2)
    return(f2-f1)/(2*dx)

## ===========================
##  Testing Code:
## ===========================

def func(x):
    return np.sin(x)

diff_result = ndiff(func,0,full=True)

# print('derivative: ', diff_result)
print('derivative: \t',diff_result[0],'\ndx: \t\t',diff_result[1],'\nerr: \t\t',diff_result[2])


      
