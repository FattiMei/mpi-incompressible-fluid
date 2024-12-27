# %%
import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
N = 25# Number of points in a side of the domain.
# Check if the matrix construction is correct.
# Discretize u = cos(x) * cos(y) * cos(z) in [0, 2pi]^3
# and its laplacian rhs = -3u.
u = np.empty(shape=(N,N,N))
rhs = np.empty(shape=(N,N,N))
h = 2*np.pi/(N -1)
# Create the manufactured solution
for i in range(N):
  for j in range(N):
    for k in range(N):
      u[i][j][k] = np.cos(h*i)*np.cos(h*j)*np.cos(h*k)
      rhs[i][j][k] = -3*u[i][j][k]
# DCT alongx, y, z
btilda = np.empty(shape=(N,N,N)) 
# dct along x
for j in range(N):
    for k in range(N):
        rhs[0][j][k]  /= np.sqrt(2)
        rhs[-1][j][k] /= np.sqrt(2)
        btilda[:][j][k] = scipy.fftpack.dct(rhs[:][j][k], type=1, norm='ortho')
# dct along y NOT DCT 1!!
rhs = btilda
for i in range(N):
    for k in range(N):
        rhs[i][0][k]  /= np.sqrt(2)
        rhs[i][-1][k] /= np.sqrt(2)
        btilda[i][:][k] = scipy.fftpack.dct(rhs[i][:][k], type=1, norm='ortho')
# dct along z NOT DCT 1!!
rhs = btilda
for i in range(N):
    for j in range(N):
        rhs[i][j][0]  /= np.sqrt(2)
        rhs[i][j][-1] /= np.sqrt(2)
        btilda[i][j][:] = scipy.fftpack.dct(rhs[i][j][:], type=1, norm='ortho')
# Compute x_tilda (Sol in dct space)
# Compute xtilde.
xtilde = np.copy(btilda)
for i in range(N):
  eigx = (2*np.cos(np.pi*(i)/(N-1))-2)/h**2
  for j in range(N):
    eigy = (2*np.cos(np.pi*(j)/(N-1))-2)/h**2
    for k in range(N):
            eigz = (2*np.cos(np.pi*(k)/(N-1))-2)/h**2
            if (i==0 and j==0 and k==0):
                xtilde[0][0][0] = 0
            else:
                xtilde[i][j][k] /= (eigx + eigy + eigz)
buff = xtilde
# Compute xhat
# dct along z
for i in range(N):
    for j in range(N):
        xtilde[i][j][:]= scipy.fftpack.idct(buff[i][j][:], type=1, norm='ortho')
        xtilde[i][j][0] *= np.sqrt(2)
        xtilde[i][j][-1]*= np.sqrt(2)
# idct along y NOT DCT 1 !!
xtilde = buff
for i in range(N):
    for k in range(N):
        buff[i][:][k]  = scipy.fftpack.idct(xtilde[i][:][k], type=1, norm='ortho')
        xtilde[i][0][k]  *= np.sqrt(2)
        xtilde[i][-1][k] *= np.sqrt(2)
# idct along x NOT DCT 1 !!
for j in range(N):
    for k in range(N):
        buff[:][j][k]  = scipy.fftpack.idct(xtilde[:][j][k], type=1, norm='ortho')
        xtilde[0][j][k]  *= np.sqrt(2)
        xtilde[-1][j][k] *= np.sqrt(2)

sol = xtilde*h**0
l = 2
plt.plot(  u[l][l][:], 'g')
plt.plot(sol[l][l][:], '-ro')
plt.show()
exit()
# %%
"""
Compute the fast laplacian for a 3D case for periodic boundary conditions on all boundaries, and check the results.
"""

## # %%
## """
## # Trivial method
## """
## 
## # %%
## # Create the matrix for the 3D Laplacian operator.
## # The unknown is stored as a vector with N^3 components, stored like in mif.
## # The rhs has the same shape.
## # Therefore, the matrix is a square N^3 x N^3 matrix.
## # To compute the matrix, an easier representation as a 6D tensor is used.
## A = np.zeros(shape=(N,N,N,N,N,N))
## for i in range(N):
##   for j in range(N):
##     for k in range(N):
##             if (i != 0 and i!= N-1):
##                 # Derivative wrt x.
##                 A[i][j][k][i][j][k] += -2
##                 A[i][j][k][(i-1)%N][j][k] += 1
##                 A[i][j][k][(i+1)%N][j][k] += 1
##             elif (i ==0):
##                 A[i][j][k][i  ][j][k] += -2
##                 A[i][j][k][i+1][j][k] +=  2
##             else :
##                 A[i][j][k][i  ][j][k] += -2
##                 A[i][j][k][i-1][j][k] +=  2
## 
##             if (j != 0 and j!= N-1):
##                 ## Derivative wrt y.
##                 A[i][j][k][i][j][k] += -2
##                 A[i][j][k][i][(j-1)%N][k] += 1
##                 A[i][j][k][i][(j+1)%N][k] += 1
##             elif (j ==0):
##                 A[i][j][k][i][j  ][k] += -2
##                 A[i][j][k][i][j+1][k] +=  2
##             else :
##                 A[i][j][k][i][j  ][k] += -2
##                 A[i][j][k][i][j-1][k] +=  2
##             if (k != 0 and k!= N-1):
##                 ## Derivative wrt z.
##                 A[i][j][k][i][j][k] += -2
##                 A[i][j][k][i][j][(k-1)%N] += 1
##                 A[i][j][k][i][j][(k+1)%N] += 1
##             elif (k ==0):
##                 A[i][j][k][i][j][k  ] += -2
##                 A[i][j][k][i][j][k+1] +=  2
##             else :                  
##                 A[i][j][k][i][j][k  ] += -2
##                 A[i][j][k][i][j][k-1] +=  2
## # reshape of the tensor in a matrix
## A = A.reshape(N**3,N**3)
## #print(A)

# %%
# Check if the matrix construction is correct.
# Discretize u = cos(x) * cos(y) * cos(z) in [0, 2pi]^3
# and its laplacian rhs = -3u.
u = np.empty(shape=(N,N,N))
rhs = np.empty(shape=(N,N,N))
h = 2*np.pi/(N)
# Create the manufactured solution
for i in range(N):
  for j in range(N):
    for k in range(N):
      u[i][j][k] = np.cos(h*i)*np.cos(h*j)*np.cos(h*k)
      rhs[i][j][k] = -3*u[i][j][k]

## # %%
## # TRY SOLVE THE SYSTEM IN THE "Dummy way"
## # set as non singular the system (ie impose the pressure in the point (0, 0, 0))
## i = 3
## A[i, :] = np.zeros(N**3)
## A[i][0] = 1
## rhs[i] = u[i]
## # Solvw the lin syst A_ij/h**2 x_j = b_i
## uh = h**2 * np.linalg.solve(A, rhs)
## # Check if the laplacian of u is rhs.
## print("N elem: ", N)
## print("\n CHECK: ", "Max : ", max(u-uh), "Min : ", min(u-uh), "check in 0 : ", u[0]-uh[0])
## print("Massima discrepanza :", np.max((A @ uh - A @ u)), "Posizione : ", np.argmax((A @ uh - A @ u)) )
## print("Minima discrepanza  :", np.min((A @ uh - A @ u)), "Posizione : ", np.argmin((A @ uh - A @ u)) )
## # Correct convergence order (2).
## plt.plot(u, '-ro')
## plt.plot(uh, '-bo')
## plt.plot(u-uh)
## plt.show()

# %%  DCT solver
# Applt dct to the whole system so we get Dx_t = b_t
bHat = np.copy(rhs)
## for i in range(N):
##     for j in range(N):
##         for k in range(N):
##             if( i == 0 or i == N-1):
##                 bHat[i][j][k] /= np.sqrt(2)
##             if( j == 0 or j == N-1):
##                 bHat[i][j][k] /= np.sqrt(2)
##             if( k == 0 or k == N-1):
##                 bHat[i][j][k] /= np.sqrt(2)
btilda = np.empty(shape=(N,N,N)) 
# dct along x
for j in range(N):
    for k in range(N):
        btilda[:][j][k] = scipy.fftpack.dct(rhs[:][j][k], type=1, norm='ortho')
# dct along y
rhs = btilda
for i in range(N):
    for k in range(N):
        btilda[i][:][k] = scipy.fftpack.dct(rhs[i][:][k], type=1, norm='ortho')
# dct along z
rhs = btilda
for i in range(N):
    for j in range(N):
        btilda[i][j][:] = scipy.fftpack.dct(rhs[i][j][:], type=1, norm='ortho')
# Compute x_tilda (Sol in dct space)
# Compute xtilde.
xtilde = np.copy(btilda)
xtilde[0,0,0] = 0
for i in range(1,N+1):
  for j in range(1,N+1):
    for k in range(1, N+1):
            if (i==1 and j==1 and k==1):
                xtilde[0][0][0] = 0
            else:
                eigx = (2*np.cos(np.pi * i / N)-2)
                eigy = (2*np.cos(np.pi * j / N)-2)
                eigz = (2*np.cos(np.pi * k / N)-2)
                xtilde[i-1][j-1][k-1] /= (eigx + eigy + eigz)
buff = xtilde
# Compute xhat
# idct along x
for j in range(N):
    for k in range(N):
        buff[:][j][k]  = scipy.fftpack.idct(xtilde[:][j][k], type=1, norm='ortho')
# idct along y
xtilde = buff
for i in range(N):
    for k in range(N):
        buff[i][:][k]  = scipy.fftpack.idct(xtilde[i][:][k], type=1, norm='ortho')
# dct along z
for i in range(N):
    for j in range(N):
        xtilde[i][j][:]= scipy.fftpack.idct(buff[i][j][:], type=1, norm='ortho')
## # Get solution (ie from xhat to x)
## xview = np.copy(xHat)
## for i in range(N):
##     for j in range(N):
##         for k in range(N):
##             if( i == 0 or i == N-1):
##                 xview[i][j][k] *= np.sqrt(2)
##             if( j == 0 or j == N-1):
##                 xview[i][j][k] *= np.sqrt(2)
##             if( k == 0 or k == N-1):
##                 xview[i][j][k] *= np.sqrt(2)
## # we now have the final solution in xview
## # Perform checks of the solution with the manufactured solution u = c(x)*c(y)*c(z) in [0, 2pi]^3
## x = xview.reshape(N**3)

l = 4
plt.plot(u[:][l][l])
plt.plot(xtilde[:][l][l])
plt.show()
