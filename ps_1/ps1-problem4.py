## PHYS512 Problem Set 1
## Minya Bai (260856843)
## ==========================
## Problem 4

# write function for polynomial, cubic spline, and rational function
# set number of points used for each method to be the same
# test the interpolation of cos(x) for each function between -pi/2 and pi/2
# repeat for Lorentzian between -1 and 1

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

## Polynomial Interpolation
def poly(func,x):
    y = func(x)

    X = np.empty([npt,npt])
    for i in range(npt):
        X[:,i]=x**i
    Xinv = np.linalg.inv(X)
    c = Xinv@y
    y_pred=X@c

    print('Error for polynomial: ',y-y_pred)
    
    return y_pred

## Cubic Spline Interpolation
def cubic_spline(func,x):                                
    y = func(x)
    xx = np.linspace(x[0],x[-1],npt)

    spln = interpolate.splrep(x,y)
    yy = interpolate.splev(xx,spln)

    print('Error for cubic spline: ',y-yy)
    
    return yy

## Rational Function Interpolation
def rat_eval(p,q,x):
    top = 0
    for i in range(len(p)):
        top = top+p[i]*x**i

    bot = 1
    for i in range(len(q)):
        bot = bot+q[i]*x**(i+1)

    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat = np.zeros([n+m-1,n+m-1])

    for i in range(n):
        mat[:,i] = x**i

    for i in range(1,m):
        mat[:,i-1+n] = -y*x**i

    pars = np.dot(np.linalg.inv(mat),y)
    p = pars[:n]
    q = pars[n:]

    return p,q

def rat_inter(func,x,n,m):
    y = func(x)
    p,q = rat_fit(x,y,n,m)
    y_pred = rat_eval(p,q,x)

    print('Error for rational functions: ',y-y_pred)
    
    return y_pred

## ==============================
##  Testing Code:
## ==============================

def func1(x):
    return np.cos(x)

def func2(x):
    return 1/(1+x**2)

# def compare_interpolation(func,npt):
n = 4
m = 5
npt = n+m-1

# For cos(x) between -pi/2 and pi/2
x1 = np.linspace(-np.pi/2,np.pi/2,npt)
x2 = np.linspace(-1,1,npt)

## Function 1 - f(x) = cos(x)
# plt.plot(x1,func1(x1),label='actual function')
# plt.plot(x1,poly(func1,x1),label='polynomial interpolation')
# plt.plot(x1,cubic_spline(func1,x1),label='cubic spline interpolation')
# plt.plot(x1,rat_inter(func1,x1,n,m),label='rational function interpolation')

## Function 2 - f(x) = 1/(1+x**2)
plt.plot(x2,func2(x2),label='actual function')
plt.plot(x2,poly(func2,x2),label='polynomial interpolation')
plt.plot(x2,cubic_spline(func2,x2),label='cubic spline interpolation')
plt.plot(x2,rat_inter(func2,x2,n,m),label='rational function interpolation')

plt.legend()
plt.show()
