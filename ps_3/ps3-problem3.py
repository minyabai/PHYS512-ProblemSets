## PHYS521 - Problem Set 3
## ===============================
## Minya Bai (260856843)
## Question 3

import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("dish_zenith.txt").T
x,y,z = data

## Part a
## ===============================
A = np.zeros((len(z),4)) # we have 4 parameters - a,b,c,d

for i in range(len(z)):
    A[i,0] = x[i]**2 + y[i]**2 # a
    A[i,1] = x[i] # b
    A[i,2] = y[i] # c
    A[i,3] = 1 # d

# Take Noise N to be I
m = np.linalg.inv(A.T@A)@A.T@z

print("The best parameters for a,b,c, and d are ",m,"respectively.")

plt.plot(z - A@m)
plt.show()

## Part b
## ===============================
noise = np.std(z - A@m)
cov = np.linalg.inv(A.T@A)/noise
print("The uncertainty in a is ", np.sqrt(cov[0][0]))

# calculate the focal length
a = m[0]
da = np.sqrt(cov[0][0])

f = 1/(4*a) / 1000 # in m
df = np.sqrt((-da/(4*a**2))**2) / 1000 # in m

print("The focal length is {} +/- {} m".format(f,df))

