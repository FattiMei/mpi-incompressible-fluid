# %%
import numpy as np
import scipy

# %%
"""
Compute the fast laplacian for a 1D case for Neumann boundary conditions on both boundaries, and check the results.
"""

# %%
N = 10 # Size of the matrix.

# %%
"""
# Trivial method
"""

# %%
# Create the matrix for the Laplacian operator.
A = [[0 for i in range(N)] for j in range(N)]

A[0][0] = -2
A[0][1] = 2

for i in range(1,N-1):
  A[i][i-1] = 1
  A[i][i] = -2
  A[i][i+1] = 1

A[N-1][N-2] = 2
A[N-1][N-1] = -2

A = np.array(A, dtype=float)
A

# %%
# Create a random solution.
np.random.seed(1)
xex = [np.random.random() for i in range(N)]
xex = np.array(xex)

# %%
# Compute the corresponding rhs.
b = A @ xex

# %%
"""
# Fast method
"""

# %%
# Compute bhat as D^-1 * b.
bhat = np.copy(b)
bhat[0] /= np.sqrt(2)
bhat[N-1] /= np.sqrt(2)

# %%
# Compute btilde.
# Documentation of the DCT options: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
btilde = scipy.fftpack.dct(bhat, type=1, norm='ortho')

# %%
# Compute xtilde.
xtilde = np.copy(btilde)
xtilde[0] = 0 # The first eigenvalue is 0: set the first value of xtilde to 0.
for i in range(1, N):
  xtilde[i] /= ((2*np.cos(np.pi * i / (N-1))-2))

# %%
# Compute xhat.
# Note that using DCT or IDCT produces the same result.
xhat = scipy.fftpack.idct(xtilde, type=1, norm='ortho')

# %%
# Compute x as D * xhat.
x = np.copy(xhat)
x[0] *= np.sqrt(2)
x[N-1] *= np.sqrt(2)

# %%
# Check the result: xex and x should differ by a constant.
tol = 1e-10
constant = xex[i] - x[i]
for i in range(1,N):
  assert xex[i] - x[i] - constant < tol

# %%
"""
# Eigenvalues and eigenvectors
"""

# %%
"""
The eigenvalues and eigenvectors are different from the ones given by the professor. The eigenvalues are the same for both A and Ahat, while the eigenvectors differ. The following code computes the eigenvalues and eigenvectors of Ahat and compares them to a closed formula I speculate to be correct and that I got from the pdf the professor put on webeep.
"""

# %%
# Compute Ahat as D^-1 * A * D.
Ahat = np.copy(A)
Ahat[0][1] = np.sqrt(2)
Ahat[1][0] = np.sqrt(2)
Ahat[N-2][N-1] = np.sqrt(2)
Ahat[N-1][N-2] = np.sqrt(2)
Ahat

# %%
# Compute the eigenvalues and eigenvectors using scipy.
eigen = np.linalg.eig(Ahat)
eigenvalues = eigen.eigenvalues
eigenvectors = eigen.eigenvectors

# %%
# Associate each eigenvalue with its eigenvector.
# Note: this is also a lazy way to sort the eigenvalues from largest to smallest.
eigen_dict = {}
for i in range(N):
  eigen_dict[eigenvalues[i]] = eigenvectors[...,i]

# %%
# Compute the expected eigenvalues.
expected_eigenvalues = [0 for i in range(N)]
for i in range(N):
  expected_eigenvalues[i] = ((2*np.cos(np.pi * i / (N-1))-2))
expected_eigenvalues = np.array(expected_eigenvalues)

# %%
# Compute the expected eigenvectors.
expected_eigenvectors = [[0 for i in range(N)] for j in range(N)]
for i in range(N):
  for j in range(N):
    expected_eigenvectors[i][j] = np.cos(np.pi * i * j / (N-1))
    if i == 0 or i == N-1:
      expected_eigenvectors[i][j] /= np.sqrt(2)

# %%
# Associate each eigenvalue with its eigenvector.
expected_eigen_dict = {}
expected_eigen_dict = {}
for i in range(N):
  expected_eigen_dict[eigenvalues[i]] = eigenvectors[...,i]

# %%
# Check if the eigenvalues are the same and the eigenvectors differ by a constant multiple.
eigenvalues = list(eigen_dict.keys())
expected_eigenvalues = list(expected_eigen_dict.keys())

tol = 1e-10
for i in range(N):
  assert abs(eigenvalues[i] - expected_eigenvalues[i]) < tol
  vec1 = eigen_dict[eigenvalues[i]]
  vec2 = expected_eigen_dict[expected_eigenvalues[i]]
  if vec2[0] == 0.0:
    multiple = 0.0
  else:
    multiple = vec1[0]/vec2[0]
  for j in range(N-1):
    assert abs(vec1[j] - multiple * vec2[j]) < tol