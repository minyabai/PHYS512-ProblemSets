## PHYS512 - Problem Set 4
## =================================
## Minya Bai (260856843)
## Code for plotting the output results

import numpy as np
from matplotlib import pyplot as plt

chain = np.loadtxt('planck_chain.txt',delimiter = ',') # chain from q3
chain2 = np.loadtxt('planck_chain_tau_prior.txt',delimiter = ',') # chain from q4

## Plotting params
## ==================================

fig1, axs = plt.subplots(6,2)
fig1.figsize = (20,8)

fig2, axs2 = plt.subplots(6,2)
fig2.figsize = (20,8)

param_names = ['$H_0$','$\Omega_bh^2$','$\Omega_Ch^2$','$\tau$','$A_s$','$n_s$']

for i in range(6):
    axs[i,0].plot(chain[:,i+1])
    axs[i,0].set_ylabel(param_names[i])
    axs[i,1].loglog(np.abs(np.fft.rfft(chain[:,i+1])))

    axs2[i,0].plot(chain2[:,i+1])
    axs2[i,0].set_ylabel(param_names[i])
    axs2[i,1].loglog(np.abs(np.fft.rfft(chain2[:,i+1])))

fig1.tight_layout()
fig2.tight_layout()


## Plotting Chisq
## =================================
# plt.plot(chain[:,0])
# plt.plot(chain2[:,0])
# plt.ylabel('$\chi^2$')

# plt.show()
