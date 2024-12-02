import os
import matplotlib.pyplot as plt

# Assumes that the main file prints exactly in this format:
# L1NORM L2NORM LINFNORM\n

# Parameters.
initial_N = 32
initial_time_steps = 1
num_tests = 4

# Remove previous results.
try:
    os.remove("results.txt")
except:
    pass

# Compile.
os.system("cd ../build && make -j")

# Run.
for step in range(num_tests):
    N = initial_N * 2**(step)
    time_steps = initial_time_steps * 2**(step)

    os.system("../build/mif " + str(N) + " " + str(time_steps) + " 1 >> results.txt")

# Get the results.
with open("results.txt", "r") as f:
    data = f.read().splitlines()

# Compute errors and speedups.
l1_speedups = []
l2_speedups = []
linf_speedups = []
speedups = [l1_speedups, l2_speedups, linf_speedups]
l1_errors = []
l2_errors = []
linf_errors = []
errors = [l1_errors, l2_errors, linf_errors]
for step in range(num_tests):
    for i in range(3):
        errors[i].append(float(data[step].split(" ")[i]))

        if i > 0:
            if (float(data[step].split(" ")[i]) != 0.0):
                speedups[i].append(float(data[step-1].split(" ")[i]) / float(data[step].split(" ")[i]))
            else:
                speedups[i].append(float("inf"))

# Plot the results.
x = []
y = []
for i in range(num_tests):
    x.append(2**i)
    y.append(1.0/2**(2*i)*5*10**-8)

# Plot the errors as a log-log plot.
# Plot the l1_errors, l2_errors, and linf_errors.
# Make sure that the known data points are marked
# as dots on the plot.
plt.figure()
plt.plot(x, l1_errors, label="L1 Norm")
plt.plot(x, l2_errors, label="L2 Norm")
plt.plot(x, linf_errors, label="Linf Norm")

# Plot the second order convergence line reference
# as a dotted line.
plt.plot(x, y, ":", label="Second order reference")

plt.scatter(x, l1_errors)
plt.scatter(x, l2_errors)
plt.scatter(x, linf_errors)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Variable-proportional deltas")
plt.ylabel("Error")
plt.legend()
plt.title("Convergence order analysis")
plt.savefig("convergence_order.png")
plt.show()