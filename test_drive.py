# test_drive_upload_oauth.py
from pathlib import Path
from mimetypes import guess_type
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CLIENT_FOLDER_ID = "1bbDtQ7WrDTyaoMTJfIlgix7QUG3is78U"  # from your Drive folder URL
LOCAL_PATH = Path(r"C:\Users\AnatKG\Documents\Neuro Science Masters\year A\python for neuroscientists\final_project\Imagination_in_translation\GT_images\wilma_ground_truth\living_room_l.jpg")

def main():
    assert LOCAL_PATH.exists() and LOCAL_PATH.stat().st_size > 0
    token = Path("token.json").read_text(encoding="utf-8")
    creds = Credentials.from_authorized_user_info(__import__("json").loads(token), SCOPES)
    service = build("drive", "v3", credentials=creds)

    meta = {"name": LOCAL_PATH.name, "parents": [CLIENT_FOLDER_ID]}
    media = MediaFileUpload(str(LOCAL_PATH), mimetype=guess_type(str(LOCAL_PATH))[0] or "application/octet-stream")
    file = service.files().create(body=meta, media_body=media, fields="id,size,webViewLink").execute()
    print("✅ Uploaded:", file["id"], "Size:", file.get("size"), "Link:", file.get("webViewLink"))

if __name__ == "__main__":
    main()
