# one_time_get_token.py
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
flow = InstalledAppFlow.from_client_secrets_file("client_secret_181198445839-k3o9vkqnv48n8anojc9mp7m8qrpld62k.apps.googleusercontent.com.json", SCOPES)
creds = flow.run_local_server(port=0)
open("token.json", "w", encoding="utf-8").write(creds.to_json())
print("Wrote token.json")