#This is not used as for now we don't have a service account
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from config import DRIVE_FOLDER


def get_drive_service(creds):
    return build("drive", "v3", credentials=creds)

def extract_folder_id_from_url(drive_url: str) -> str:
    """Extract folder ID from Google Drive share URL"""
    # Handle different URL formats:
    # https://drive.google.com/drive/folders/1ABC123DEF456?usp=drive_link
    # https://drive.google.com/drive/folders/1ABC123DEF456
    if "/folders/" in drive_url:
        folder_id = drive_url.split("/folders/")[1].split("?")[0].split("/")[0]
        return folder_id
    else:
        raise ValueError(f"Invalid Google Drive folder URL: {drive_url}")

def create_folder(service, name, parent_id=None):
    """Modified to work with shared folder as default parent"""
    # If no parent_id provided, use the shared folder as parent
    if parent_id is None:
        parent_id = DRIVE_FOLDER
    
    # Check if folder already exists
    query = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id)").execute()
    items = results.get("files", [])
    if items:
        print(f"✅ Found existing folder: {name} (ID: {items[0]['id']})")
        return items[0]["id"]
    
    # Create new folder
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id]  # Fixed: removed duplicate parents assignment
    }
    
    try:
        file = service.files().create(body=file_metadata, fields="id").execute()
        print(f"✅ Created folder: {name} (ID: {file['id']}) in parent: {parent_id}")
        return file["id"]
    except Exception as e:
        print(f"❌ Failed to create folder '{name}' in parent '{parent_id}': {e}")
        raise

def upload_file(service, file_path: Path, mime_type: str, parent_id: str):
    """Upload file to specified folder"""
    try:
        file_metadata = {"name": file_path.name, "parents": [parent_id]}
        media = MediaFileUpload(str(file_path), mimetype=mime_type)
        uploaded = service.files().create(body=file_metadata, media_body=media, fields="id,webViewLink").execute()
        print(f"✅ Uploaded: {file_path.name} (ID: {uploaded['id']})")
        return uploaded["id"]
    except Exception as e:
        print(f"❌ Failed to upload {file_path.name}: {e}")
        raise
