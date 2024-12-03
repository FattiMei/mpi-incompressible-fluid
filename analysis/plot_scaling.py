import os
import matplotlib.pyplot as plt
import time

# Parameters.
N = 60
time_steps = 40
P_max = 6

# Remove previous results.
try:
    os.remove("results.txt")
except:
    pass

# Compile.
os.system("cd ../build && make -j >/dev/null")

# Run (convergence over x).
Px_times = []
for P in range(1, P_max+1):
    before = time.time_ns()
    os.system("mpirun -n " + str(P) + " ../build/mif " + str(N) + " " + str(time_steps) + " " + str(P) + " >/dev/null")
    after = time.time_ns()
    Px_times.append((after-before) / (N ** 3 * time_steps))

# Run (convergence over y).
Py_times = []
for P in range(1, P_max+1):
    before = time.time_ns()
    os.system("mpirun -n " + str(P) + " ../build/mif " + str(N) + " " + str(time_steps) + " 1 >/dev/null")
    after = time.time_ns()
    Py_times.append((after-before) / (N ** 3 * time_steps))

# Plot the results.
x = []
y = []
for i in range(1,P_max+1):
    x.append(i)
    y.append(Px_times[0] / i)

plt.figure()
plt.plot(x, Px_times, label="Scaling over x")
plt.plot(x, Py_times, label="Scaling over y")
plt.plot(x, y, ":", label="Ideal scaling")

plt.scatter(x, Px_times)
plt.scatter(x, Py_times)
plt.xlabel("P")
plt.ylabel("Time (ns) per point, time step")
plt.legend()
plt.title("Scaling analysis")
plt.savefig("scaling.png")
plt.show()