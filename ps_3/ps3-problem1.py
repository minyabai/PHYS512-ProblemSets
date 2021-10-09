## PHYS521 - Problem Set 3
## ===============================
## Minya Bai (260856843)
## Question 1

import numpy as np
from matplotlib import pyplot as plt

def f(x,y):
    dydx = y/(1+x**2)
    return dydx

def rk4_step(fun,x,y,h):
    k1 = fun(x,y) 
    k2 = fun(x+h/2, y+h*k1/2)
    k3 = fun(x+h/2, y+h*k2/2)
    k4 = fun(x+h, y+h*k3)
    dy = h*(k1 + 2*k2 + 2*k3 + k4)/6
    return y+dy

def rk4_stepd(fun,x,y,h): # taken from numerical recipes (pg. 911)
    full = rk4_step(fun,x,y,h)
    step1 = rk4_step(fun,x,y,h/2)
    step2 = rk4_step(fun,x+h/2,step1,h/2)

    return step2 + (step2 - full)/15 

npt1 = 601 # needs x3 more data points to make same number of function evals
npt2 = 201
x1 = np.linspace(-20,20,npt1)
x2 = np.linspace(-20,20,npt2)

y1 = np.zeros(npt1)
y1[0] = 1

y2 = np.zeros(npt2)
y2[0] = 1

# since x is equally space, just calculate it once before for loop
h1 = x1[1] - x1[0]
h2 = x2[1] - x2[0]

for i in range(npt1-1):
    y1[i+1] = rk4_step(f,x1[i],y1[i],h1)

for i in range(npt2-1):
    y2[i+1] = rk4_stepd(f,x2[i],y2[i],h2)

# Calculate deviation from truth
i1 = y1[-1] - y1[0]
i2 = y2[-1] - y2[0]
actual = truth1[-1] - truth1[0]
    
print("Actual Value: ", actual)
print("Single Step: {} deviates from actual by {}".format(i1, actual-i1))
print("Two Steps: {} deviates from actual by {}".format(i2, actual-i2))

# Find the coefficient of the truth function
c0 = 1/np.exp(np.arctan(-20))
truth1 = c0 * np.exp(np.arctan(x1))
truth2 = c0 * np.exp(np.arctan(x2))

# plt.ion()
# plt.plot(x1,y1)
# plt.plot(x1,truth1)
plt.plot(x1,np.abs(y1-truth1),label = 'rk4_step')
plt.plot(x2,np.abs(y2-truth2),label = 'rk4_stepd')
plt.legend()
plt.show()
