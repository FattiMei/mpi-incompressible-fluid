import matplotlib.pyplot as plt
import numpy as np
# Load the data from the file
filename = "../MaxErrDirich.txt"

# Open the file and read the data
with open(filename, 'r') as file:
    data = file.readlines()

# Convert data to a list of numbers (strip newline characters and convert to float)
values = [float(line.strip()) for line in data]

# Plot the data
plt.figure(figsize=(10, 5))  # Optional: set the figure size
plt.loglog(values, marker='o', linestyle='-', color='b')  # Customize plot style
plt.loglog(np.arange(len(values), dtype=np.float32)**-2, '-or')
plt.title("Plot of Values from File")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)  # Optional: add a grid
plt.show()
