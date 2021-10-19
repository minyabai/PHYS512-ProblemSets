## PHYS 512 - Problem Set 4
## =============================
## Minya Bai (260856843)
## Main file for running code

import numpy as np
import camb
from matplotlib import pyplot as plt
import time
from functions import *

## Question 1
## ==================================
pars=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3]);
model=get_spectrum(pars)
model=model[:len(spec)]
resid=spec-model
chisq=np.sum((resid/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.")

## read in a binned version of the Planck PS for plotting purposes
# planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
# errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3]);
# plt.clf()
# plt.plot(ell,model)
# plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
# plt.show()

## Question 2
## =================================

threshold = 0.01
deltas = 0.05 * pars
derivs, model = derivative(pars,deltas,spec)
resid = spec-model
chisq = np.sum((resid/errs)**2)
lmb = 0

n = 10

for i in range(n):
    lhs = (derivs.T/errs**2)@(derivs)
    lhs = lhs + lmb*np.diag(np.diag(lhs))
    rhs = (derivs.T/errs**2) @ (resid)

    dm = np.linalg.inv(lhs)@rhs

    new_pars = pars + dm
    new_derivs, new_model = derivative(new_pars, deltas, spec)
    new_resid = spec - new_model
    new_chisq = np.sum((new_resid/errs)**2)

    d_chisq = new_chisq - chisq

    if new_chisq  < chisq:
        pars, resid, derivs, chisq = new_pars, new_resid, new_derivs, new_chisq
        lmb = update_lmb(lmb, success = True)
        if lmb == 0:
            if (abs(d_chisq) < threshold):
                print(pars)
                break

    else:
        lmb = update_lmb(lmb, success = False)

cov = np.linalg.inv(lhs)
pars_err = np.sqrt(np.diag(cov)) # calculate uncertainty in values

params_data = np.asarray((pars,pars_err)).T # fit param data
np.savetxt('planck_fit_params.txt', params_data, delimiter=',') # save the data

## Question 3
## ===============================

pars  = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95]) # re-initiate starting vals
scale = 0.5 # scale of step
nstep = 3000 # no. steps

chain = run_mcmc(spec,pars,nstep,errs,cov) # run mcmc
np.savetxt('planck_chain.txt', chain, delimiter = ',') # save the data

# To avoid re-running code for mcmc, the plotting is done with different
# python file that will read the saved data and analyze it

chain_chisq = chain[:,0] # chisq vals
chain_params = chain[:,1:] # param vals

# calculate Omega_lambda

params_new = np.zeros(6) # mean param value
std_new = np.zeros(6) # error of param

for i in range(6):
    params_new[i] = np.mean(chain[:,i+1])
    std_new[i] = np.std(chain[:,i+1])

h = params_new[0]/100
dh = std_new[0]/100

h2 = h**2
dh2 =	h2*(2*dh/h)

Omg_b =	params_new[1]/(h**2)
dOmg_b = np.sqrt((std_new[1]/params_new[1])**2+(dh2/h2)**2)

Omg_C = params_new[2]/(h**2)
dOmg_C = np.sqrt((std_new[2]/params_new[2])**2+(dh2/h2)**2)

Omg_l =	1 - Omg_b - Omg_C
dOmg_l = np.sqrt(dOmg_b**2 + dOmg_C**2)

print(Omg_l, dOmg_l)

## Question 4
## =================================

tau = chain[:,4] # gets tau values
tau_weights = gauss(tau) # calculates weights from gaussian

w_cov = np.cov(chain_params.T,aweights=tau_weights) # new cov using weights

chain2 = run_mcmc(spec,pars,nstep,errs,w_cov,r_tau=True) # rerun chain
np.savetxt('planck_chain_tau_prior.txt',chain2,delimiter=',')

# let's compare the vals from second chain and first chain!

param_names = ['H_0','O_bh^2','O_Ch^2','tau','A_s','n_s']

print("======================================================================")

for i in range(1,len(pars)):
    isp = np.sum(tau_weights*chain[:,i])/np.sum(tau_weights)
    print("{} import sampled param from Q3 : \t{}".format(param_names[i-1],isp))
    print("{} Param from new chain : \t\t {}".format(param_names[i-1],np.mean(chain2[:,i])))
    print("======================================================================")
