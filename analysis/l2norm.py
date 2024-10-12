import scipy
import numpy as np
import sympy as sp
from sympy.abc import x, y, z
import matplotlib.pyplot as plt


def vertex_mean_quadrature_formula(U, Lx, Ly, Lz):
    # assumes a regular mesh, doesn't bother about performant loops
    Nx, Ny, Nz = U.shape
    acc = 0

    for i in range(Nx-1):
        for j in range(Ny-1):
            for k in range(Nz-1):
                acc += sum([
                    U[i  , j  , k  ],
                    U[i+1, j  , k  ],
                    U[i  , j+1, k  ],
                    U[i+1, j+1, k  ],
                    U[i  , j  , k+1],
                    U[i+1, j  , k+1],
                    U[i  , j+1, k+1],
                    U[i+1, j+1, k+1]
                ]) / 8

    return acc * (Lx / (Nx-1)) * (Ly / (Ny-1)) * (Lz / (Nz-1))


Lx = 1
Ly = 2
Lz = 0.9


if __name__ == '__main__':
    # symbolic_integrand = x*x + x*sp.cos(y)*sp.exp(sp.sin(z))
    symbolic_integrand = (1 + sp.exp(-x)) * sp.sin(y*z)
    concrete_integrand = sp.lambdify([x,y,z], symbolic_integrand)

    exact_integral = sp.integrate(
        symbolic_integrand,
        (x, 0, Lx),
        (y, 0, Ly),
        (z, 0, Lz)
    ).evalf()

    nodes = 2 ** np.arange(1,8)
    err = np.zeros(len(nodes))

    for i, n in enumerate(nodes):
        XX, YY, ZZ = np.meshgrid(
            np.linspace(0, Lx, n),
            np.linspace(0, Ly, n),
            np.linspace(0, Lz, n)
        )

        solution_points = concrete_integrand(XX,YY,ZZ)
        approx_integral = vertex_mean_quadrature_formula(
            solution_points,
            Lx,
            Ly,
            Lz
        )

        err[i] = abs(exact_integral - approx_integral) / exact_integral

    plt.title('Convergence behaviour of 3D quadrature formula')
    plt.xlabel('divisions')
    plt.ylabel('relative error')

    plt.loglog(nodes, err, label='mean vertex')
    plt.loglog(nodes, 1/nodes**2, label='$O(N^{-2})$')
    plt.legend()

    plt.show()
