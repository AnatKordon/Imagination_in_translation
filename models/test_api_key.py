
#before using it install:
#pip install requests python-dotenv
#pip install python-dotenv
#and create a file ".env" with our api key in this format: STABILITY_API_KEY= (and then api key without quotes)
#set your ".env" file under .gitignore so that you won't share the secret api key!!!

from dotenv import load_dotenv
import os

load_dotenv()  # This reads .env and sets the variables

api_key = os.getenv("STABILITY_API_KEY")

if api_key:
    print("✅ API key loaded:", api_key[:6] + "..." + api_key[-4:])
else:
    print("❌ API key not found. Did you create a .env file?")
