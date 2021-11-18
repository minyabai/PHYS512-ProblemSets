## PHYS512 - Problem Set 7
## ==========================
## Minya Bai (260856843)
## Question 1

import numpy as np
from matplotlib import pyplot as plt

x1,y1,z1 = np.loadtxt("rand_points.txt").T
x2,y2,z2 = np.loadtxt("rand_points_np.txt").T

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(x1,y1,z1,'.',s=0.1)
plt.show()

ax.clear()
ax.scatter(x2,y2,z2,'.',s=0.1)
plt.show()

