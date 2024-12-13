import matplotlib.pyplot as plt
import numpy as np
# Load the data from the file
filenameDirichlet = "../MaxErrDirich.txt"
filenameNeumann = "../MaxErrNeumann.txt"

# Open the file and read the data
with open(filenameNeumann, 'r') as file:
    dataNeumann = file.readlines()
with open(filenameDirichlet, 'r') as file:
    dataDirichlet = file.readlines()

# Convert data to a list of numbers (strip newline characters and convert to float)
valuesDirichlet = [float(line.strip()) for line in dataDirichlet]
valuesNeumann = [float(line.strip()) for line in dataNeumann]

# Plot the data
plt.loglog(valuesDirichlet, marker='o', linestyle='-', color='b')  # Customize plot style
plt.loglog(valuesNeumann, marker='o', linestyle='-', color='r')  # Customize plot style
plt.loglog(np.arange(len(valuesNeumann), dtype=np.float32)**-2, '-og')
plt.title("Plot of Values from File")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)  # Optional: add a grid
plt.show()
