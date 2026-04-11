import pandas as pd
import os

# 1. Setup your paths
path_a = r"D:\Thesis\Repository\Dynamic-Bus-Routing-for-On-Demand-Public-Transportation-\results"
path_b = r"D:\Thesis\Repository\Dynamic-Bus-Routing-for-On-Demand-Public-Transportation-\results ( greedy + v3ant)"
output_dir = r"D:\Thesis\Repository\Dynamic-Bus-Routing-for-On-Demand-Public-Transportation-\Combined_Flat_Files"

# 2. List of benchmark folders from your screenshot
folders = [
    "baseline", "capacity_8", "capacity_24", "demand_busy", "demand_quiet",
    "fleet_4", "fleet_8", "maxwait_15", "maxwait_45", "ridefactor_15", "ridefactor_30"
]

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

for folder in folders:
    file_path_a = os.path.join(path_a, folder, "aggregated.csv")
    file_path_b = os.path.join(path_b, folder, "aggregated.csv")
    
    # Verify both files exist before attempting merge
    if os.path.exists(file_path_a) and os.path.exists(file_path_b):
        print(f"Processing: {folder}...")
        
        df_a = pd.read_csv(file_path_a)
        df_b = pd.read_csv(file_path_b)
        
        # Merge the dataframes
        combined_df = pd.concat([df_a, df_b], ignore_index=True)
        
        # Define the new flat filename (e.g., baseline_results.csv)
        new_filename = f"{folder}_results.csv"
        final_save_path = os.path.join(output_dir, new_filename)
        
        # Save to the flat directory
        combined_df.to_csv(final_save_path, index=False)
        print(f"  -> Saved as: {new_filename}")
    else:
        # Check which one is missing for easier debugging
        if not os.path.exists(file_path_a): print(f"  [!] Missing file in Path A for: {folder}")
        if not os.path.exists(file_path_b): print(f"  [!] Missing file in Path B for: {folder}")

print("\nTask Complete. Your merged files are in:", output_dir)