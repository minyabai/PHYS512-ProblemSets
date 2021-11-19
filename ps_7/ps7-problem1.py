## PHYS512 - Problem Set 7
## ==========================
## Minya Bai (260856843)
## Question 1

import numpy as np
from matplotlib import pyplot as plt

x1,y1,z1 = np.loadtxt("rand_points.txt").T
x2,y2,z2 = np.loadtxt("rand_points_np.txt").T

a = 5.5
b = -3

fig, ax = plt.subplots(1,2)

ax[0].set_title('Using C rand int generator')
ax[0].plot(a*x1+b*y1, z1, '.', markersize=0.5)

ax[1].set_title('Using np rand int generator')
ax[1].plot(a*x2+b*y2, z2, '.', markersize=0.5)
plt.show()



