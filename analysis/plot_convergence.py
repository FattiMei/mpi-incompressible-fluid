import os
from math import log2
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
os.system("cd ../build && cmake .. &&make -j")

# Run.
for step in range(num_tests):
    N = initial_N * 2**(step)
    time_steps = initial_time_steps * 2**(step)
    print("Running test with N = ", N, " and time_steps = ", time_steps)
    os.system("../build/mif " + str(N) + " " + str(time_steps) + " >> results.txt")

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
    y.append(1.0/2**(2*i)*linf_errors[0])

plt.loglog(x, l1_errors, label="L1")
plt.loglog(x, l2_errors, label="L2")
plt.loglog(x, linf_errors, label="Linf")
plt.loglog(x, y, label="Delta^2")

plt.title("Convergence order")
plt.xlabel("Delta")
plt.ylabel("Error")
plt.legend()
plt.savefig("convergence_order.png")
plt.show()

# now time the execution of the code
import time

start = time.time()
for i in range(10):
    os.system("../build/mif " + str(N / 2) + " " + str(time_steps / 2) + " >> results.txt")
end = time.time()

print("Execution time: ", (end - start) / 10, "s")
