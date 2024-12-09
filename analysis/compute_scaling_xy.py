import os
import matplotlib.pyplot as plt
import time

# Parameters.
N = 60
time_steps = 40
P_max = 8  # must be a power of 2

# Compile.
os.system("cd ../build && make -j >/dev/null")

# Compute possible numbers of processors.
P_list = [1]
while P_list[-1] < P_max:
    P_list.append(P_list[-1] * 2)

# Run (convergence over xy).
times = []
for i in range(len(P_list)):
    Px = P_list[i]
    Py = P_max // Px
    before = time.time_ns()
    os.system(
        "mpirun -n "
        + str(P_max)
        + " ../build/mif "
        + str(N)
        + " "
        + str(time_steps)
        + " "
        + str(Px)
        + " >/dev/null"
    )
    after = time.time_ns()
    times.append(((Px, Py), after - before))

# Run (serial).
before = time.time_ns()
os.system("../build/mif " + str(N) + " " + str(time_steps) + " 1 >/dev/null")
after = time.time_ns()
serial_time = after - before

for value in times:
    Px, Py = value[0]
    P_time = value[1]
    print(f"Px: {Px}, Py: {Py}, time: {P_time}ns, speedup: {serial_time/P_time}")
