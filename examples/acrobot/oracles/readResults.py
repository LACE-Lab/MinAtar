import numpy as np
import glob

# Specify the path pattern for the results files
file_name = "2023_07_05_acrobot_one_state_oracle_h4.dim.4.t.0.5"
results_file_pattern = f"weights/{file_name}.*.results"

# Get a list of all results file paths matching the pattern
results_files = glob.glob(results_file_pattern)

# Accumulate values from each file
total_weight1 = []
total_weight2 = []
total_weight3 = []
total_weight4 = []

for file_path in results_files:
    # Load the data from the results file, skipping the header row
    data = np.loadtxt(file_path, delimiter="\t", skiprows=1)

    # Extract column -7 (weight1) and column -6 (weight2) and accumulate the values
    weight1 = data[:, -11]
    weight2 = data[:, -10]
    weight3 = data[:, -9]
    weight4 = data[:, -8]
    total_weight1.extend(weight1)
    total_weight2.extend(weight2)
    total_weight3.extend(weight3)
    total_weight4.extend(weight4)

# Calculate the average for weight1 and weight2
average_weight1 = np.mean(total_weight1)
average_weight2 = np.mean(total_weight2)
average_weight3 = np.mean(total_weight3)
average_weight4 = np.mean(total_weight4)

output_file_name = f"{file_name}_weights"

# Write the final results to the output file
with open(output_file_name, "w") as f:
    f.write(str(average_weight1) + "\t" + str(average_weight2) + "\t" + str(average_weight3) + "\t" + str(average_weight4) + "\n")

# Print the final averages
print("Average of weight1 (column -7):", average_weight1)
print("Average of weight2 (column -6):", average_weight2)
print("Average of weight3 (column -6):", average_weight3)
print("Average of weight4 (column -6):", average_weight4)
