import pandas as pd
import os
import glob
from tabulate import tabulate

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
df = read_all_csv_files(r'C:\Users\yaniv\Github repositories\Imagination_in_Translation\Imagination_in_translation\ui\data\logs')

# Display basic info about the combined DataFrame
if not df.empty:
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
else:
    print("No data loaded")
## For wide tables, transpose or select specific columns:
print("Transposed view (first 5 rows):")
print(tabulate(df.head().T, headers='keys', tablefmt='grid'))

# Or select specific columns
#columns_to_show = ['column1', 'column2', 'column3']  # Replace with your actual column names
#print(tabulate(df[columns_to_show].head(), headers='keys', tablefmt='grid'))
