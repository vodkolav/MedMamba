import os
import gdown

# Ensure FILE_ID and FILENAME are set
file_id = os.getenv("FILE_ID")
filename = os.getenv("FILENAME")

if not file_id or not filename:
    raise ValueError("FILE_ID and FILENAME environment variables must be set")

# Construct the Google Drive URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
gdown.download(url, filename, quiet=False)