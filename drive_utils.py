# helpers_drive.py - for use with token.json
from pathlib import Path
import os
import json
from io import BytesIO
from mimetypes import guess_type
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


FOLDER_MIME = "application/vnd.google-apps.folder"

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# getting the token either locally or from st.secrets
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


def ensure_folder(service, name: str, parent_id: str) -> str:
    """Return the folder id named `name` under `parent_id`, creating it if missing.
    Name of folder can be the user id, the sessuib_id or the gen_images folder..."""
    q = (
        f"name = '{name}' and mimeType = '{FOLDER_MIME}' "
        f"and '{parent_id}' in parents and trashed=false"
    )
    res = service.files().list(q=q, fields="files(id,name)", pageSize=1).execute()
    items = res.get("files", [])
    if items:
        return items[0]["id"]
    meta = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
    created = service.files().create(body=meta, fields="id").execute()
    return created["id"]

def ensure_path(service, parent_id: str, *parts: str) -> str:
    """Create a nested folder path under parent_id, return the deepest folder id."""
    cur = parent_id
    for part in parts:
        cur = ensure_folder(service, part, cur)
    return cur

def update_or_insert_file(service, file_path: Path, parent_id: str, dest_name: str | None = None,
                mime_type: str | None = None) -> str:
    """
    Create or replace a file by name under parent_id.
    If a file with `dest_name` exists in that folder (not a folder), update it; else create.
    Returns file id.
    """
    file_path = Path(file_path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty file: {file_path}")

    name = dest_name or file_path.name
    mime = mime_type or (guess_type(str(file_path))[0] or "application/octet-stream")
    q = (
        f"name = '{name}' and '{parent_id}' in parents "
        f"and mimeType != '{FOLDER_MIME}' and trashed=false"
    )
    res = service.files().list(q=q, fields="files(id,name)", pageSize=1).execute()
    media = MediaFileUpload(str(file_path), mimetype=mime, resumable=False)

    if res.get("files"):
        file_id = res["files"][0]["id"]
        service.files().update(fileId=file_id, media_body=media, fields="id").execute()
        return file_id
    else:
        meta = {"name": name, "parents": [parent_id]}
        created = service.files().create(body=meta, media_body=media, fields="id").execute()
        return created["id"]


def extract_folder_id(url_or_id: str) -> str:
    if "drive.google.com" in url_or_id and "/folders/" in url_or_id:
        return url_or_id.split("/folders/")[1].split("?")[0].split("/")[0]
    return url_or_id


def build_drive_from_token_dict(token_dict):
    creds = Credentials.from_authorized_user_info(token_dict, DRIVE_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("drive", "v3", credentials=creds)

# def upload_path_to_folder(service, local_path: Path, folder_id: str):
#     local_path = Path(local_path)
#     assert local_path.exists() and local_path.stat().st_size > 0
#     meta = {"name": local_path.name, "parents": [folder_id]}
#     media = MediaFileUpload(str(local_path), mimetype=mimetypes.guess_type(str(local_path))[0] or "application/octet-stream")
#     return service.files().create(body=meta, media_body=media, fields="id,size,webViewLink").execute()

# def upload_bytes_to_folder(service, data: bytes, name: str, folder_id: str, mime: str = "application/octet-stream"):
#     meta = {"name": name, "parents": [folder_id]}
#     media = MediaIoBaseUpload(BytesIO(data), mimetype=mime, resumable=False)
#     return service.files().create(body=meta, media_body=media, fields="id,size,webViewLink").execute()




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

