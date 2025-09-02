# cloud_storage.py
from __future__ import annotations
from typing import Optional
from pathlib import Path
from mimetypes import guess_type
import json

# --- Google OAuth Drive backend ---
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from google.oauth2.credentials import Credentials
from io import BytesIO

def _drive_creds_from_secrets(google_client: dict, google_token: dict):
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    creds = Credentials.from_authorized_user_info(google_token, SCOPES)
    # Refresh if expired and we have a refresh_token
    if creds and creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request
        creds.refresh(Request())
    return creds

def drive_init(st_secrets) -> "DriveClient":
    if "google_client" not in st_secrets or "google_token" not in st_secrets:
        raise RuntimeError("Missing st.secrets['google_client'] or st.secrets['google_token']")
    creds = _drive_creds_from_secrets(st_secrets["google_client"], st_secrets["google_token"])
    service = build("drive", "v3", credentials=creds)
    return DriveClient(service)

class DriveClient:
    def __init__(self, service): self.service = service
    def upload_file(self, local_path: Path, folder_id: str, name: Optional[str] = None) -> str:
        local_path = Path(local_path)
        if not local_path.exists() or local_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty: {local_path}")
        meta = {"name": name or local_path.name, "parents": [folder_id]}
        media = MediaFileUpload(str(local_path), mimetype=guess_type(str(local_path))[0] or "application/octet-stream")
        file = self.service.files().create(body=meta, media_body=media, fields="id,size,webViewLink").execute()
        return file["id"]
    def upload_bytes(self, data: bytes, dest_name: str, folder_id: str, mime: str = "application/octet-stream") -> str:
        meta = {"name": dest_name, "parents": [folder_id]}
        media = MediaIoBaseUpload(BytesIO(data), mimetype=mime, resumable=False)
        file = self.service.files().create(body=meta, media_body=media, fields="id,size,webViewLink").execute()
        return file["id"]
