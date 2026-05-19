"""
AI Matte — Minimal Colab notebook (3 cells).
Canonical copy: keep in sync with google_desk_app/colab_templates/
"""

# CELL 1
from google.colab import auth, drive
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
import sys

CODE_FILE_ID = 'YOUR_CODE_FILE_ID_HERE'

print("=" * 60)
print("AI Matte — SAM3 + BiRefNet + ViTMatte")
print("=" * 60)
auth.authenticate_user()
drive_service = build('drive', 'v3')
code_path = '/content/cellcode.py'
request = drive_service.files().get_media(fileId=CODE_FILE_ID)
with open(code_path, 'wb') as f:
    downloader = MediaIoBaseDownload(f, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"   Downloading: {int(status.progress() * 100)}%")
sys.path.insert(0, '/content')
with open(code_path, 'r', encoding='utf-8') as _cf:
    _head = _cf.read(4000)
if 'setup_ai_matte_cell1' not in _head:
    raise RuntimeError(
        'Wrong CODE_FILE_ID: file is not AI Matte cellcode. '
        'Re-run Send to Colab from the Desk app (or use VDA_Jobs/code/<job_id>_cellcode.py from that job).'
    )
from cellcode import setup_ai_matte_cell1
drive_service = setup_ai_matte_cell1(drive_service)
print("\n[OK] Cell 1 complete.")


# CELL 2
CONFIG_FILE_ID = 'YOUR_CONFIG_FILE_ID_HERE'
from cellcode import load_ai_matte_config_cell2
config, drive_base_path, date_folder, logger, is_batch, config_path = load_ai_matte_config_cell2(
    drive_service, CONFIG_FILE_ID)
print(f"\n[OK] Config ready — batch={is_batch}")


# CELL 3
from cellcode import process_ai_matte_cell3
_ok = process_ai_matte_cell3(
    config, drive_base_path, date_folder, logger, is_batch, config_path)
if _ok:
    print("\n[DONE] AI Matte complete.")
else:
    print("\n[STOP] Fix errors above, then re-run Cell 3.")
