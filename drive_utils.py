# helpers_drive.py - for use with token.json
from pathlib import Path
from mimetypes import guess_type
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# helpers_drive.py (drop this in, or inline in your app)
import os, json, mimetypes
from io import BytesIO
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def extract_folder_id(url_or_id: str) -> str:
    if "drive.google.com" in url_or_id and "/folders/" in url_or_id:
        return url_or_id.split("/folders/")[1].split("?")[0].split("/")[0]
    return url_or_id


def get_token_dict_from_secrets_or_env(st):
   # 1) Path to token.json - this is what i have
    p = os.getenv("GOOGLE_TOKEN_PATH")
    if p and Path(p).exists():
        return json.loads(Path(p).read_text(encoding="utf-8"))

    # 2) JSON string in env
    j = os.getenv("GOOGLE_TOKEN")
    if j:
        try:
            return json.loads(j)
        except json.JSONDecodeError as e:
            raise RuntimeError("GOOGLE_TOKEN env var is not valid JSON") from e

    if st is not None:
        try:
            return st.secrets["google_token"]
        except Exception:
            pass

    raise RuntimeError(
        "No Google token found. Set GOOGLE_TOKEN_FILE=<path to token.json> "
        "or GOOGLE_TOKEN=<token.json contents>, or provide [google_token] in st.secrets."
    )
    raise RuntimeError("No Google token found. Add [google_token] to st.secrets or set env GOOGLE_TOKEN to the token JSON.")

def build_drive_from_token_dict(token_dict):
    creds = Credentials.from_authorized_user_info(token_dict, DRIVE_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("drive", "v3", credentials=creds)

def upload_path_to_folder(service, local_path: Path, folder_id: str):
    local_path = Path(local_path)
    assert local_path.exists() and local_path.stat().st_size > 0
    meta = {"name": local_path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(local_path), mimetype=mimetypes.guess_type(str(local_path))[0] or "application/octet-stream")
    return service.files().create(body=meta, media_body=media, fields="id,size,webViewLink").execute()

def upload_bytes_to_folder(service, data: bytes, name: str, folder_id: str, mime: str = "application/octet-stream"):
    meta = {"name": name, "parents": [folder_id]}
    media = MediaIoBaseUpload(BytesIO(data), mimetype=mime, resumable=False)
    return service.files().create(body=meta, media_body=media, fields="id,size,webViewLink").execute()




# #This is not used as for now we don't have a service account
# from pathlib import Path
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaFileUpload
# from config import DRIVE_FOLDER


# def get_drive_service(creds):
#     return build("drive", "v3", credentials=creds)

# def extract_folder_id_from_url(drive_url: str) -> str:
#     """Extract folder ID from Google Drive share URL"""
#     # Handle different URL formats:
#     # https://drive.google.com/drive/folders/1ABC123DEF456?usp=drive_link
#     # https://drive.google.com/drive/folders/1ABC123DEF456
#     if "/folders/" in drive_url:
#         folder_id = drive_url.split("/folders/")[1].split("?")[0].split("/")[0]
#         return folder_id
#     else:
#         raise ValueError(f"Invalid Google Drive folder URL: {drive_url}")

# def create_folder(service, name, parent_id=None):
#     """Modified to work with shared folder as default parent"""
#     # If no parent_id provided, use the shared folder as parent
#     if parent_id is None:
#         parent_id = DRIVE_FOLDER
    
#     # Check if folder already exists
#     query = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder'"
#     if parent_id:
#         query += f" and '{parent_id}' in parents"
#     results = service.files().list(q=query, fields="files(id)").execute()
#     items = results.get("files", [])
#     if items:
#         print(f"✅ Found existing folder: {name} (ID: {items[0]['id']})")
#         return items[0]["id"]
    
#     # Create new folder
#     file_metadata = {
#         "name": name,
#         "mimeType": "application/vnd.google-apps.folder",
#         "parents": [parent_id]  # Fixed: removed duplicate parents assignment
#     }
    
#     try:
#         file = service.files().create(body=file_metadata, fields="id").execute()
#         print(f"✅ Created folder: {name} (ID: {file['id']}) in parent: {parent_id}")
#         return file["id"]
#     except Exception as e:
#         print(f"❌ Failed to create folder '{name}' in parent '{parent_id}': {e}")
#         raise

# def upload_file(service, file_path: Path, mime_type: str, parent_id: str):
#     """Upload file to specified folder"""
#     if not file_path.exists():
#         raise FileNotFoundError(f"File not found: {file_path}")
#     if not parent_id:
#         raise ValueError("No parent folder ID provided")
#     try:
#         parent_info = service.files().get(fileId=parent_id, fields="id,name").execute()
#         print(f"📁 Uploading {file_path.name} to folder: {parent_info.get('name')}")

#         file_metadata = {"name": file_path.name, "parents": [parent_id]}
#         media = MediaFileUpload(str(file_path), mimetype=mime_type)
#         uploaded = service.files().create(body=file_metadata, media_body=media, fields="id,webViewLink").execute()
#         print(f"✅ Uploaded: {file_path.name} (ID: {uploaded['id']})")
#         return uploaded["id"]
#     except Exception as e:
#         print(f"❌ Upload failed for {file_path.name}: {e}")
#         # Log detailed error info
#         if "403" in str(e):
#             print("💡 Permission denied - check folder sharing")
#         elif "404" in str(e):
#             print("💡 Folder not found - check folder ID")
#         elif "quota" in str(e).lower():
#             print("💡 Storage quota exceeded")
#         raise

