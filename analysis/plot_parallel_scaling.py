import os
import matplotlib.pyplot as plt
import time

# Parameters.
N = 240
time_steps = 50
P_max = 6


# Update input file.
def update_input(N, time_steps, Py, Pz):
    data = (
        "Nt : "
        + str(time_steps)
        + "\ndt : 1e-3\n"
        + "Nx : "
        + str(N)
        + "\n"
        + "Ny : "
        + str(N)
        + "\n"
        + "Nz : "
        + str(N)
        + "\n"
        + "Py : "
        + str(Py)
        + "\n"
        + "Pz : "
        + str(Pz)
        + "\ntest_case_2 : false"
    )

    with open("../input/input.txt", "w") as f:
        f.write(data)


# Compile.
os.system("cd ../build && make -j >/dev/null")

# Run (convergence over z).
Pz_times = []
for P in range(1, P_max + 1):
    update_input(N, time_steps, 1, P)
    before = time.time_ns()
    os.system("mpirun -n " + str(P) + " ../build/mif ../input/input.txt")
    after = time.time_ns()
    Pz_times.append(after - before)

# Run (convergence over y).
Py_times = [Pz_times[0]]
for P in range(2, P_max + 1):
    update_input(N, time_steps, P, 1)
    before = time.time_ns()
    os.system("mpirun -n " + str(P) + " ../build/mif ../input/input.txt")
    after = time.time_ns()
    Py_times.append(after - before)

# Remove .vtk and .dat files.
os.system("rm *vtk && rm *dat")

# Compute speedups
Pz_speedups = [Pz_times[0] / p_time for p_time in Pz_times]
Py_speedups = [Py_times[0] / p_time for p_time in Py_times]

# Plot the results.
x = []
y = []
for i in range(1, P_max + 1):
    x.append(i)
    y.append(i)

plt.figure()
plt.plot(x, Pz_speedups, label="Scaling over z")
plt.plot(x, Py_speedups, label="Scaling over y")
plt.plot(x, y, ":", label="Ideal scaling")

plt.scatter(x, Pz_speedups)
plt.scatter(x, Py_speedups)
plt.xlabel("P")
plt.ylabel("Speedup")
plt.legend()
plt.title("Scaling analysis")
plt.savefig("scaling.png")
plt.show()
