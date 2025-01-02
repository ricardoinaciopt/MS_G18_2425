import pandas as pd

# Load the dataset
file_path = "simulation_results.csv"  # Replace with the correct file path
simulation_data = pd.read_csv(file_path)


# Map 'sex' column: F -> 0, M -> 1
if "sex" in simulation_data.columns:
    simulation_data["sex"] = simulation_data["sex"].map({"F": 0, "M": 1})
    print("\nUpdated 'sex' column:")
    print(simulation_data["sex"].unique())

if "group" in simulation_data.columns:
    simulation_data["group"] = simulation_data["group"].map({"A": 1, "B": 0})
    print("\nUpdated 'group' column:")
    print(simulation_data["group"].unique())

# # Replace 'False' with 0 and 'True' with 1 in boolean columns
# boolean_columns = [
#     "opportunities", "has_disease", "has_car", "has_house", "job_status", "personal_luxuries"
# ]

# for column in boolean_columns:
#     if column in simulation_data.columns:
#         simulation_data[column] = simulation_data[column].replace({False: 0, True: 1})
#         print(f"\nUpdated '{column}' column:")
#         print(simulation_data[column].unique())

# Save the updated dataset in the same file
simulation_data.to_csv(file_path, index=False)
print(f"\nUpdated dataset saved to {file_path}.")
