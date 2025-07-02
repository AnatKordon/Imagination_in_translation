from dotenv import load_dotenv
import os
load_dotenv()

#this is another option if storing key in .env file
client_secret_path = os.getenv('GOOGLE_CLIENT_SECRET_PATH')


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


# Auth
gauth = GoogleAuth()
gauth.LoadClientConfigFile(client_secret_path) #this part is missing in Yaniv's code
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

print('success!')