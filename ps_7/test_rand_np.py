import numpy as np
from matplotlib import pyplot as plt

n = 100000000
vec = np.random.randint(0,2**31,n*3)

vv = np.reshape(vec,[n,3])
vmax = np.max(vv,axis=1)

maxval = 1e8
vv2 = vv[vmax<maxval,:]


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(vv2[0],vv2[1],vv2[2],'.',s=0.1)
# plt.show()



f = open('rand_points_np.txt','w')
for i in range(vv2.shape[0]):
    myline = repr(vv2[i,0])+' '+repr(vv2[i,1])+' '+repr(vv2[i,2])+'\n'
    f.write(myline)
f.close()
