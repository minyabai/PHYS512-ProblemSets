## PHYS512 - Problem Set 2
## Minya Bai (260856843)
## ==========================
## Question 1
##
## We want to calculate the E field from an infinitessimally thin spherical shell. We can used Gauss's
## Law to get estimate of what the E field should be outside the charge. When z < R, since there is no
## charge inside the shell, E = 0. When z >= R, we can integrate to find the efield.


import numpy as np
from legendre_integrate import integrate
from scipy.integrate import quad
from matplotlib import pyplot as plt

# z > R
def outside(x,z):
    a = p*R**2/2*eps_0

    numer = (z-R*np.cos(x))*np.sin(x)
    denom = (R**2+z**2-2*R*z*np.cos(x))**(3/2)

    return a * numer/denom

# z < R
def cal_efield(z): # z-position
    params=[z]
    xmin = 0
    xmax = np.pi
    dx_targ = 0.1

    if z < R:
        return 0.0,0.0
        
    else:
        targ = p*R**2/(z**2*eps_0)
        integrator = integrate(outside,xmin,xmax,dx_targ,5,params)
        quad_int = quad(outside,xmin,xmax,args=(z))[0]

        #return targ
        
    return integrator,quad_int
        
## ============================
## Testing Code
## =============================
# constants
eps_0 = 1
R=5
p=1

# print("Efield = ",cal_efield(10))

r = np.linspace(0,50,100) # There is runtime error at z=R using integrator method
E_i,E_q = np.array([cal_efield(i) for i in r]).T

plt.plot(r,E_i,'.',label='Integrator Method')
plt.plot(r,E_q,'.',label='Quad Method')
plt.legend()
plt.show()
