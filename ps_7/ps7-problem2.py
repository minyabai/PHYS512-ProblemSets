## PHYS512 - Problem Set 7
## ============================
## Minya Bai (260856843)
## Question 2

import numpy as np
from matplotlib import pyplot as plt

def gauss(x,sigma):
    return np.exp(-x**2/(2*sigma**2))

def lorentz(x):
    return 1/(1+x**2)

def power(x,a,b,c):
    return (x+a)**b + c

## Taking a look at the exp in comparision to the possible options

# x = np.linspace(0,5,1000)

# plt.plot(x, gauss(x,0.5), label = 'Gaussian')
# plt.plot(x, lorentz(x), label = 'Lorentzian')
# plt.plot(x, power(x,0.5,-0.5,0), label = 'Power')
# plt.plot(x, np.exp(-x), label = 'Exponential')
# plt.legend()
# plt.show()

## Lorentzian is best option for the probability density function
## since all the points are above the exponential
## Let y=int lorentz = atan(x), so x = tan(y) for y \in (-\pi/2,\pi/2)
## So P(y) = P(tan(y))

n = 1000000
y = np.pi*(np.random.randn(n)-0.5)
dev = np.tan(y)
dev_bound = dev[np.abs(dev)<25]

a,bins = np.histogram(dev_bound, 100)
binC = (bins[1:]+bins[:-1])/2

pred = lorentz(binC)
pred = pred/pred.sum()
a = a/a.sum()

## Plot comparison between lorentz and deviate
# plt.plot(binC, a/a.max(), label='histogram')
# plt.plot(binC, pred/pred.max(), label='lorentzian')
# plt.legend()
# plt.show()

dev_cut = dev[dev > 0] # we only want non-negative exp dev
p = 1.0 # scaling
cut_prob = p * np.exp(-dev_cut)/lorentz(dev_cut)
cut = np.random.rand(len(cut_prob)) < cut_prob
deviates = dev_cut[cut]

print("Max Accepted fraction: ", np.max(cut_prob))
print("Accepted fraction: ", len(deviates)/len(dev_cut))

expdev = np.random.exponential(p,len(deviates))

deviates.sort()
expdev.sort()

## Plotting Histogram
plt.hist(deviates, bins=200, density=True)
plt.hist(expdev, bins=200, density=True)
plt.show()
