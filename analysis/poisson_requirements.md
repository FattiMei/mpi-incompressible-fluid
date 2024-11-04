# Poisson requirements
Given a 1D Poisson problem defined on the interval $[0,L]$

$$
    \nabla^2 p(x) = f(x)
$$

Implement a direct solver using the FastPoisson method for the boundary conditions:
  1. $ \frac{\partial p}{\partial x} = 0$ on the boundary
  2. $p(0) = p(L)$


## Requirements

### Matrix building
 * build the finite difference matrix $A$ for your problem (dense matrix should be good, you could try sparse matrices).

No implementation of H and Psi matrices (the matrices of the right and left eigenvectors respectively) is required


### Achievements
  * create a custom solution $x$
  * $b = Ax$
  * $\tilde{b} = DCT(b)$ this particular transform is compatible with the laplacian operator and the boundary conditions
  * compute $\tilde{x}$ by dividing $\tilde{b}$ with the eigenvalues of the operator (available on webeep)
  * get back $x$ with the "inverse" operation of the DCT
  * compare with the original solution


### Status
  * @FattiMei: periodic, doesn't work
