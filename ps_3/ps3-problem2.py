## PHYS521 - Problem Set 3
## ===============================
## Minya Bai (260856843)
## Question 2

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

## Part a
## ==============================
# conversions to years
y_d = 365
y_h = 24 * y_d
y_m = 60 * y_h
y_s = 60 * y_m
y_us = 1e6 * y_s

## Using the chart from the U238 Decay, we get the following half life in years
hl_U238 = [4.468e9, 24.10/y_d, 6.70/y_h, 245500, 75380, 1600, 3.8235/y_d,
             3.10/y_m, 26.8/y_m, 19.9/y_m, 164.2/y_us, 22.3, 5.015,
             138.376/y_d]

def fun(x,y,half_life=hl_U238):
    dydx = np.zeros(len(half_life)+1)

    #first and last case are different
    dydx[0] = -y[0]/half_life[0] * np.log(2)
    dydx[-1] = y[-2]/half_life[-2] * np.log(2)

    for i in range(1,len(half_life)): # for loop does to len-1
        dydx[i] = y[i-1]/half_life[i-1] - y[i]/half_life[i]

    return dydx

y0 = np.zeros(len(hl_U238)+1) # Keeping track of products
y0[0] = 1 # start with just U238
x0 = 0
x1 = 1e10 # arbitrary large end time

ans_stiff = integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')

## Part b
## ==============================
plt.plot(ans_stiff.t, ans_stiff.y[-1]/ans_stiff.y[0], '.')
# plt.plot(ans_stiff.t, ans_stiff.y[4]/ans_stiff.y[3], '.')
plt.xlabel("Time [years]")
plt.ylabel("Pb206/U238")
# plt.ylabel("Th230/U238")
plt.xscale('log')
plt.show()





