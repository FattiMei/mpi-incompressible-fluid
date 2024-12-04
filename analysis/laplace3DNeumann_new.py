import numpy as np
import scipy.fftpack  # Import fftpack specifically
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 100  # Number of points in a side of the domain.

# Discretize u = cos(x) * cos(y) * cos(z) in [0, 2pi]^3
# and its Laplacian rhs = -3u.
u = np.empty(shape=(N, N, N))
rhs = np.empty(shape=(N, N, N))
h = 2 * np.pi / (N - 1)

# Create the manufactured solution
for i in range(N):
    for j in range(N):
        for k in range(N):
            u[i, j, k] = np.cos(h * i) * np.cos(h * j) * np.cos(h * k)
            rhs[i, j, k] = -3 * u[i, j, k]

# DCT along x, y, z
btilda = np.empty(shape=(N, N, N))

# DCT along x
for j in range(N):
    for k in range(N):
        # Corrected indexing from [:][j][k] to [:, j, k]
        # Removed manual scaling of boundary elements
        btilda[:, j, k] = scipy.fftpack.dct(rhs[:, j, k], type=2, norm='ortho')  # Changed DCT type to 2

# DCT along y
for i in range(N):
    for k in range(N):
        # Corrected indexing from [i][:][k] to [i, :, k]
        btilda[i, :, k] = scipy.fftpack.dct(btilda[i, :, k], type=2, norm='ortho')  # Changed DCT type to 2

# DCT along z
for i in range(N):
    for j in range(N):
        # Corrected indexing from [i][j][:] to [i, j, :]
        btilda[i, j, :] = scipy.fftpack.dct(btilda[i, j, :], type=2, norm='ortho')  # Changed DCT type to 2

# Compute x_tilda (Solution in DCT space)
xtilde = np.copy(btilda)
for i in range(N):
    eigx = (2 * np.cos(np.pi * i / (N - 1)) - 2) / h ** 2
    for j in range(N):
        eigy = (2 * np.cos(np.pi * j / (N - 1)) - 2) / h ** 2
        for k in range(N):
            eigz = (2 * np.cos(np.pi * k / (N - 1)) - 2) / h ** 2
            if i == 0 and j == 0 and k == 0:
                xtilde[0, 0, 0] = 0  # Handle the zero eigenvalue case
            else:
                xtilde[i, j, k] /= (eigx + eigy + eigz)

# Inverse DCT along z
for i in range(N):
    for j in range(N):
        # Corrected indexing and removed manual scaling
        xtilde[i, j, :] = scipy.fftpack.idct(xtilde[i, j, :], type=2, norm='ortho')  # Changed IDCT type to 2

# Inverse DCT along y
for i in range(N):
    for k in range(N):
        # Corrected indexing and removed manual scaling
        xtilde[i, :, k] = scipy.fftpack.idct(xtilde[i, :, k], type=2, norm='ortho')  # Changed IDCT type to 2

# Inverse DCT along x
for j in range(N):
    for k in range(N):
        # Corrected indexing and removed manual scaling
        xtilde[:, j, k] = scipy.fftpack.idct(xtilde[:, j, k], type=2, norm='ortho')  # Changed IDCT type to 2

# Final solution
sol = xtilde

# Plotting
# l = 0
# plt.plot(u[l, l, :], 'g', label='Analytical Solution')
# plt.plot(sol[l, l, :], '-ro', label='Numerical Solution')
# plt.legend()
#plt.surf()
# Surface plot
# Choose a fixed z-plane (e.g., middle of the domain)
k_fixed = N // 2  # Index of the fixed z-plane

# Create meshgrid for x and y coordinates
x = np.linspace(0, 2 * np.pi, N)
y = np.linspace(0, 2 * np.pi, N)
X, Y = np.meshgrid(x, y)

# Extract the slice at the fixed z-plane for both solutions
u_slice = u[:, :, k_fixed]
sol_slice = sol[:, :, k_fixed]

# Plotting the analytical solution
fig = plt.figure(figsize=(14, 6))

# Analytical solution surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, u_slice - sol_slice, cmap='viridis')
ax1.set_title('Analytical Solution at z = {:.2f}'.format(h * k_fixed))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x, y, z)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# # Numerical solution surface plot
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# surf2 = ax2.plot_surface(X, Y, sol_slice, cmap='viridis')
# ax2.set_title('Numerical Solution at z = {:.2f}'.format(h * k_fixed))
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_zlabel('sol(x, y, z)')
# fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()

plt.show()