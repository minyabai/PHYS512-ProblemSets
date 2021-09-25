## PHYS 512 - Problem Set 2
## ===========================
## Minya Bai (260856843)
## Question 2

import numpy as np

## example from class
## ===========================

def integrate_adaptive_class(fun,x0,x1,tol):
    # print('integrating between ',x0,x1)
    #hardwire to use simpsons                                                                                                            
    x=np.linspace(x0,x1,5)
    y=fun(x) # want to reduce num times this is called
    dx=(x1-x0)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step                                                                                         
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step                                                                              
    err=np.abs(area1-area2)
    if err<tol:
        return area2
    else:
        xmid=(x0+x1)/2
        left=integrate_adaptive_class(fun,x0,xmid,tol/2)
        right=integrate_adaptive_class(fun,xmid,x1,tol/2)
        return left+right

## recursive implementation
## ===========================

# quadratic Simpson's method
def quad_simpson(fun,x0,f0,x1,f1):
    m = (x0 + x1) / 2 # finds mid point
    fm = fun(m) # calculates func at mid point

    return m, fm, abs(x1 - x0) / 6 * (f0 + 4 * fm + f1)

def integrate_adaptive(fun,x0,x1,tol,extra=None):
    # print('integrating between ',x0,x1)
    if extra == None: # equivalent to initial function call
        f0 = fun(x0)
        f1 = fun(x1)

        # calculates the middle pt, the func value at middle point, and
        # the total area using simpson's method with those three points
        xm, fm, whole = quad_simpson(fun,x0,f0,x1,f1)

        return integrate_adaptive(fun,x0,x1,tol,(f0,f1,xm,fm,whole))

    else: # recursive function
        f0, f1, xm, fm, whole = extra # unpack extra from previous region

        # calculates the area to the left and right of the middle point
        xlm, flm, left = quad_simpson(fun,x0,f0,xm,fm)
        xrm, frm, right = quad_simpson(fun,xm,fm,x1,f1)

        # left + right is equivalently taking a finer 5 pt area and whole is with just the three
        err = (left + right) - whole

        if abs(err) <= tol: # base case
            return left + right
        else: # otherwise
            lextra = (f0,fm,xlm,flm,left) # extra for left side
            rextra = (fm,f1,xrm,frm,right) # extra for right side

            return integrate_adaptive(fun,x0,xm,tol/2,lextra) + integrate_adaptive(fun,xm,x1,tol/2,rextra)

## ============================
## Testing Code
## ============================

x0 = 0
x1 = 1

# r_count = 0

actual = -np.cos(1)+np.cos(0)
recur_int = integrate_adaptive(np.sin,0,1,1e-09)
class_int = integrate_adaptive_class(np.sin,0,1,1e-09)

print(actual)
print(recur_int)

print("Recursive Integrator gives {} with error {}".format(recur_int, abs(recur_int-actual))) 
print("Integrator from class gives {} with error {}".format(class_int, abs(class_int-actual)))
