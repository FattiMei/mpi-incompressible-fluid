import os
import matplotlib.pyplot as plt

# Assumes that the main file prints exactly in this format:
# L1NORM L2NORM LINFNORM\n

# Parameters.
initial_N = 24
initial_time_steps = 1
num_tests = 5

# Remove previous results.
try:
    os.remove("results.txt")
except:
    pass

# Compile.
os.system("cd ../build && make -j >/dev/null")

# Run.
for step in range(num_tests):
    N = initial_N * 2 ** (step)
    time_steps = initial_time_steps * 2 ** (step)

    os.system(
        "mpirun -n 6 ../build/full_test "
        + str(N)
        + " "
        + str(time_steps)
        + " 3 >> results.txt"
    )

# Get the results.
with open("results.txt", "r") as f:
    data = f.read().splitlines()

# Compute errors.
l1_errors_v = []
l2_errors_v = []
linf_errors_v = []
l1_errors_p = []
l2_errors_p = []
linf_errors_p = []
errors = [
    l1_errors_v,
    l2_errors_v,
    linf_errors_v,
    l1_errors_p,
    l2_errors_p,
    linf_errors_p,
]
for step in range(num_tests):
    for i in range(6):
        errors[i].append(float(data[step].split(" ")[i]))

# Plot the results.
x = []
y1 = []
y2 = []
for i in range(num_tests):
    x.append(2**i)
    y1.append(1.0 / 2 ** (1.5 * i) * linf_errors_p[0])
    y2.append(1.0 / 2 ** (2 * i) * linf_errors_v[0])

# Plot the errors as a log-log plot.
# Plot the l1_errors, l2_errors, and linf_errors.
# Make sure that the known data points are marked
# as dots on the plot.
plt.figure()
plt.plot(x, l1_errors_v, label="Velocity error L1 Norm")
plt.plot(x, l2_errors_v, label="Velocity error L2 Norm")
plt.plot(x, linf_errors_v, label="Velocity error Linf Norm")
plt.plot(x, l1_errors_p, label="Pressure error L1 Norm")
plt.plot(x, l2_errors_p, label="Pressure error L2 Norm")
plt.plot(x, linf_errors_p, label="Pressure error Linf Norm")

# Plot the 1-2 order convergence line references
# as dotted lines.
plt.plot(x, y1, ":", label="Order 1.5 reference")
plt.plot(x, y2, ":", label="Order 2 reference")

plt.scatter(x, l1_errors_v)
plt.scatter(x, l2_errors_v)
plt.scatter(x, linf_errors_v)
plt.scatter(x, l1_errors_p)
plt.scatter(x, l2_errors_p)
plt.scatter(x, linf_errors_p)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Variable-proportional deltas")
plt.ylabel("Error")
plt.legend()
plt.title("Convergence order analysis")
plt.savefig("convergence_order.png")
plt.show()
