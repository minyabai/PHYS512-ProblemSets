## code taken from class                                                                                                        

import numpy as np
import legendre

def integrate(fun,xmin,xmax,dx_targ,ord=2,params=None,verbose=False):
    coeffs=legendre.integration_coeffs_legendre(ord+1)
    if verbose: #should be zero  
        print("fractional difference between first/last coefficients is "+repr(coeffs[0]/coeffs[-1]-1))

    npt=int((xmax-xmin)/dx_targ)+1
    nn=(npt-1)%(ord)
    if nn>0:
        npt=npt+(ord-nn)
    assert(npt%(ord)==1)
    npt=int(npt)

    x=np.linspace(xmin,xmax,npt)
    dx=np.median(np.diff(x))
    dat=fun(x,*params)

    #we could have a loop here, but note that we can also reshape our data, then som along columns, and only then                
    #apply coefficients.  Some care is required with the first and last points because they only show up once.                   
    mat=np.reshape(dat[:-1],[(npt-1)//(ord),ord]).copy()
    mat[0,0]=mat[0,0]+dat[-1] #as a hack, we can add the last point to the first                                                 
    mat[1:,0]=2*mat[1:,0] #double everythin in the first column, since each element appears as the last element in the previous row
    
    vec=np.sum(mat,axis=0)
    tot=np.sum(vec*coeffs[:-1])*dx
    return tot
