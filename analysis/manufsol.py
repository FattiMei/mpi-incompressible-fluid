import sympy as sp
from sympy.core import diff
t, x, y, z, Re = sp.symbols('t x y z Re')
'''
Needed to insert the manufactured solution (Both u as vector and P).
- Check that the flow is incompressible
 + Otherwise gives a valid element for w (Why? Because I decided so.)
- Gives the forcing term to insert in the solver to do the check of convergence
'''
# Definition of functions u, P
u = x**3
v = x**3
w = x**3
P = x*y
# Re = 10e+5

def laplacian(u):
    return ( sp.diff( u , x, 2) + sp.diff( u , y, 2) + sp.diff( u , z, 2) )

def manufsol(u, v, w, P):
    ## div(u) = 0
    # CHECK incomprimibilità
    if (sp.simplify( sp.diff(u, x, 1) + sp.diff(v, y, 1) + sp.diff(w, z, 1) ) !=0 ):
      print("The proposed manufactured solution is NOT good for an incompressible FLOW!!")
      w = -z*sp.simplify(sp.diff(u, x, 1) + sp.diff(v, y, 1))  
      print("Here's a good function for w: ", w)
        
    ## du/dt + dot(u,grad(u)) = - grad(p) + laplacian(u)/Re + f 
    fx = sp.simplify( sp.diff(u, t, 1) + u*diff(u, x, 1) +   v*diff(u, y, 1) +   w*diff(u, z, 1) + diff(P, x, 1) - laplacian(u)/Re )
    fy = sp.simplify(sp.diff(v, t, 1) + u*diff(v, x, 1) +   v*diff(v, y, 1) +   w*diff(v, z, 1) + diff(P, y, 1) - laplacian(v)/Re )
    fz = sp.simplify(sp.diff(w, t, 1) + u*diff(w, x, 1) +   v*diff(w, y, 1) +   w*diff(w, z, 1) + diff(P, z, 1) - laplacian(w)/Re) 
    return print( "fx: ", fx, "\n fy: ", fy, "\n fz: ", fz, )

manufsol(u, v, w, P)
