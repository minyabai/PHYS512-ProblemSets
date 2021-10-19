## functions for OAproOAblemOA setAOAOA
import numpy as np
import camb
import time

def get_spectrum(pars,lmax=3000): # gets spectrum from params
    #print('pars are ',pars)                                                                            
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE            

    return tt[2:]

def derivative(pars, delta, spec): # calculates the jacobian matrix as a function of params
    H0, omgbh2, omgCh2, tau, As, ns = pars
    d_H0, d_omgbh2, d_omgCh2, d_tau, d_As, d_ns = delta

    model = get_spectrum(pars)[:len(spec)]
    
    jacobian = np.zeros((len(model),len(pars)))
    jacobian[:,0] = (get_spectrum([H0+d_H0,omgbh2,omgCh2,tau,As,ns])[:len(model)]-model)/d_H0
    jacobian[:,1] = (get_spectrum([H0,omgbh2+d_omgbh2,omgCh2,tau,As,ns])[:len(model)]-model)/d_omgbh2
    jacobian[:,2] = (get_spectrum([H0,omgbh2,omgCh2+d_omgCh2,tau,As,ns])[:len(model)]-model)/d_omgCh2
    jacobian[:,3] = (get_spectrum([H0,omgbh2,omgCh2,tau+d_tau,As,ns])[:len(model)]-model)/d_tau
    jacobian[:,4] = (get_spectrum([H0,omgbh2,omgCh2,tau,As+d_As,ns])[:len(model)]-model)/d_As
    jacobian[:,5] = (get_spectrum([H0,omgbh2,omgCh2,tau,As,ns+d_ns])[:len(model)]-model)/d_ns

    return jacobian, model

def update_lmb(lamda,success):
    if success:
        lamda = lamda/1.5
        if lamda < 0.5:
            lamby = 0
    else:
        if lamda==0:
            lamda=1
        else:
            lamda=lamda*1.5**2
    return lamda

def steps(cov): # calculates step size using cholesky 
    cholesky = np.linalg.cholesky(cov)
    step = cholesky@np.random.randn(len(cholesky))
    return step

def prior_chisq(pars,pars_prior,pars_errs): # calculates prior of pars from chisq
    if pars_prior is None:
        return 0
    par_shifts = pars - pars_errs
    return np.sum((par_shifts/pars_errs)**2)

def gauss(x): # gaussian defined by the tau parameter
    x0 = 0.0540
    sig = 0.0074
    amp = 1/np.sqrt(2*np.pi*sig**2)
    return amp*np.exp(-0.5*(x-x0)**2/sig**2)

def is_weight(vals, w): # calculates importance sampling weight
    mean = np.average(vals, weights = w)
    std = np.sqrt(np.average((vals-mean) ** 2, weights = w))
    return mean, std

def run_mcmc(data,start_params,nstep,errs,cov,r_tau=False,scale=None,pars_priors=None,pars_errs=None):
    accepted_steps = 0 # this is just to keep track of the # of accepted runs
    nparam = len(start_params)
    chain = np.zeros([nstep,nparam+1])
    chain[0,1:] = start_params # first params is the initial params
    resid = data-get_spectrum(start_params)[:len(data)]
    cur_chisq = np.sum((resid/errs)**2) + prior_chisq(start_params,pars_priors,pars_errs) 
    chain[0,0] = cur_chisq # chisq in first column
    cur_params = start_params.copy()
    if scale == None:
        scale = np.ones(nparam)
    for i in range(nstep-1):
        if i%100 == 0:
            print("Complete {} Steps!".format(i))
            print("Accepted {} Steps!".format(accepted_steps))
        trial_params = cur_params + scale * steps(cov)

        if r_tau: # if we are constraining tau
            trial_params[3] = np.random.normal(0.0540,0.0074) # samples tau from gaussian
        
        trial_resid = data - get_spectrum(trial_params)[:len(data)]
        trial_chisq = np.sum((trial_resid/errs)**2)+prior_chisq(cur_params,pars_priors,pars_errs)
        if trial_chisq < cur_chisq:
            accept = True
            accepted_steps += 1
        else:
            delta_chisq = trial_chisq - cur_chisq
            prob = np.exp(-0.5*delta_chisq)
            if np.random.rand() < prob:
                accept = True
                accepted_steps += 1
            else:
                accept = False
        if accept:
            cur_params = trial_params
            cur_chisq = trial_chisq
        chain[i+1,1:] = cur_params
        chain[i+1,0] = cur_chisq
    return chain
