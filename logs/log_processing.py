import pandas as pd
import os
import glob
from tabulate import tabulate
import matplotlib.pyplot as plt

def read_all_csv_files(directory_path="."):
    """
    Read all CSV files from a directory and combine them into a single DataFrame
    
    Parameters:
    - directory_path: Path to directory containing CSV files (default is current directory)
    
    Returns:
    - Combined pandas DataFrame
    """
    # Get all CSV files in the directory
    csv_pattern = os.path.join(directory_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return pd.DataFrame()
    
    # Read and combine all CSV files
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add a column to track which file the data came from
            df['source_file'] = os.path.basename(file)
            dataframes.append(df)
            print(f"Successfully read: {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dataframes:
        # Combine all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nCombined {len(dataframes)} CSV files into DataFrame with {len(combined_df)} rows")
        return combined_df
    else:
        return pd.DataFrame()

# Read all CSV files from current directory (logs folder)
df = read_all_csv_files(r'.\ui\data\logs')

avg_similarity = df.groupby(['uid', 'session'])['similarity'].mean()

print("\nAverage similarity score per uid per session:")
print(tabulate(avg_similarity.reset_index(), headers='keys', tablefmt='grid'))

# Create line graph for each uid
plt.figure(figsize=(12, 8))
    
# Get unique UIDs
unique_uids = df['uid'].unique()

for uid in unique_uids:
    # Get data for this specific uid
    uid_data = avg_similarity[avg_similarity.index.get_level_values('uid') == uid]
    # Extract sessions and similarity values
    sessions = uid_data.index.get_level_values('session')
    similarities = uid_data.values
    # Plot line for this uid
    plt.plot(sessions, similarities, marker='o', label=f'UID: {uid}', linewidth=2)

# Add plot formatting AFTER the loop
plt.xlabel('Session Number')
plt.ylabel('Average Similarity Score')
plt.title('Average Similarity Score Across Sessions by UID')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
        
