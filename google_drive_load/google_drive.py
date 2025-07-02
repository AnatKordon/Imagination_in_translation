import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

# Auth
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Opens browser for login
drive = GoogleDrive(gauth)

def read_csv_as_dataframe(file_title:str) -> pd.DataFrame :
    """read_csv_as_dataframe _summary_
    read a CSV file and return it as a pd.DataFrame
    
    Parameters
    ----------
    file_title : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    file_list = drive.ListFile({'q': f"title='{file_title}'"}).GetList()

    if file_list:
        gfile = file_list[0]
        
        # Download and read as DataFrame
        gfile.GetContentFile(file_title)
        df = pd.read_csv(file_title)
        
        return df
    else:
        print(f"File '{file_title}' not found!")
        return None


def upload_dataframe_to_drive(df: pd.DataFrame, filename: str, file_format: str = 'csv'):
    """
    Upload a pandas DataFrame to Google Drive
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to upload
    filename : str
        Name for the file on Google Drive (without extension)
    file_format : str
        Format to save the file ('csv', 'json', 'excel')
    """
    
    # Add appropriate file extension
    if file_format == 'csv':
        full_filename = f"{filename}.csv"
        df.to_csv(full_filename, index=False)
    elif file_format == 'json':
        full_filename = f"{filename}.json"
        df.to_json(full_filename, orient='records', indent=2)
    elif file_format == 'excel':
        full_filename = f"{filename}.xlsx"
        df.to_excel(full_filename, index=False)
    else:
        raise ValueError("file_format must be 'csv', 'json', or 'excel'")
    
    # Upload to Google Drive
    gfile = drive.CreateFile({'title': full_filename})
    gfile.SetContentFile(full_filename)
    gfile.Upload()
    
    # Clean up local file
    #os.remove(full_filename)
    
    print(f"DataFrame uploaded as '{full_filename}' to Google Drive!")

def upload_file_to_drive(file_to_upload:str):
    """upload_file_to_drive 
    Uploads a file to Google Drive with the given file name.

    Parameters
    ----------
    file_to_upload : str
  
    """
    
    gfile = drive.CreateFile({'title': file_to_upload})
    gfile.SetContentFile(file_to_upload)
    gfile.Upload()
    print(f"Uploaded '{file_to_upload}' to Google Drive!")

# Create a sample DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NYC', 'LA', 'Chicago']
})

df_from_google =  read_csv_as_dataframe('log_file.csv')

# Upload as CSV
upload_dataframe_to_drive(df, 'user_data', 'csv')

# Upload as JSON
upload_dataframe_to_drive(df, 'user_data', 'json')

# Upload as Excel
upload_dataframe_to_drive(df, 'user_data', 'excel')

upload_file_to_drive('log_file_to_upload.txt')
