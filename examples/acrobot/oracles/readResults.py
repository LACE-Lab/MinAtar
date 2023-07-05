import numpy as np

# Specify the path to the results file
results_file_path = "test.results"

# Load the data from the results file
data = np.loadtxt(results_file_path, delimiter="\t")

# Extract column 3 and calculate the average
column_3 = data[:, 2]
average_column_3 = np.mean(column_3)

# Print the average
print("Average of column 3:", average_column_3)
