from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def get_drive_service(creds):
    return build("drive", "v3", credentials=creds)

def create_folder(service, name, parent_id=None):
    query = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id)").execute()
    items = results.get("files", [])
    if items:
        return items[0]["id"]
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder"
    }
    if parent_id:
        file_metadata["parents"] = [parent_id]
    file = service.files().create(body=file_metadata, fields="id").execute()
    return file["id"]

def upload_file(service, file_path: Path, mime_type: str, parent_id: str):
    file_metadata = {"name": file_path.name, "parents": [parent_id]}
    media = MediaFileUpload(str(file_path), mimetype=mime_type)
    uploaded = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return uploaded["id"]
