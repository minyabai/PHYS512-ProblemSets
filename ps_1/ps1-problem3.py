## PHYS512 Problem Set 1
## Minya Bai (260856843)
## ============================
## Problem 3

# As shown in the pdf, the trend between points seems fairly linear. So to simplify the problem,
# I will interpolate the data using linear interpolation. To estimate V, we need to find the two
# data points (V1, V2) such that V1 < V < V2. To find V1 and V2, I will use the most straight
# forward method of just iterating through a sorted list and looking for the first V1 s.t V1 < V.
# To approximate uncertainty is the standard deviation of V from taking different combinations
# of the 4 points nearest to V.

import numpy as np
from matplotlib import pyplot as plt

def lakeshore(V,data):
    # Assume it can either be int or array
    if isinstance(V,type(list)): 
        T = [find_neighbours(i,data) for i in V]
    else: 
        T = find_neighbours(V,data)

    return T

def find_neighbours(V,data):
    V_data = data[1][::-1] # Want as function of V, so reverse data so it goes from low to high in V
    T_data = data[0][::-1] # Also just assume the data is already in order by magnitude
    
    # Use Linear Interpolation
    n = 0

    while True:
        if V < V_data[n]: # looking for the first data point greater V
            break

        n = n + 1
        
    V1 = V_data[n]
    V2 = V_data[n+1]
    T1 = T_data[n]
    T2 = T_data[n+1]
    
    dT = (V-V1) * (T2-T1)/(V2-V1)
    T = T1+dT # best linearly interpolated value 
    
    # Approx error - T if we took different points near T
    T_all = []
    T_all.append(T)
    T_all.append(T_data[n]+(V-V_data[n]) * (T_data[n+2]-T_data[n])/(V_data[n+2]-V_data[n]))
    T_all.append(T_data[n-1]+(V-V_data[n-1]) * (T_data[n+1]-T_data[n-1])/(V_data[n+1]-V_data[n-1]))
    
    err = np.std(T_all) # standard deviation of the values 

    return [T, err]
    
## ============================
##  Testing Code:
## ============================

data=np.loadtxt('lakeshore.txt').T # load data and transpose it

print(lakeshore([1.5,1.6],data))

## Plotting raw data
# plt.plot(data[1],data[0],'.')
# plt.xlabel('Voltage [V]')
# plt.ylabel('Temperature [K]')
# plt.show()
