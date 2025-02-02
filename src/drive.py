import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2 import service_account
from core.config import settings

# 서비스 계정 인증 파일 경로
SERVICE_ACCOUNT_FILE = os.path.expanduser("~/.config/gspread/service_account.json")

# Google Drive API 인증
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/drive"]
)
service = build("drive", "v3", credentials=creds)

print("Google Drive API 인증 성공")

def download_image_from_drive(file_id, file_name, download_path="images"):
    """Google Drive에서 이미지를 로컬로 다운로드하는 함수"""
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(download_path, file_name)

    os.makedirs(download_path, exist_ok=True)  # 다운로드 폴더 생성

    with open(file_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Downloading {file_name}: {int(status.progress() * 100)}% complete.")

    return file_path

# Google Drive에서 이미지 가져오기
def get_images_from_drive():
    images = []
    # 로컬 다운로드 디렉토리
    download_dir = "images"

    continents = [
        ("Africa", settings.BEFORE_AFRICA),
        ("Europe", settings.BEFORE_EUROPE),
        ("Asia", settings.BEFORE_ASIA),
        ("North America", settings.BEFORE_NORTH_AMERICA),
        ("Oceania", settings.BEFORE_OCEANIA),
        ("South America", settings.BEFORE_SOUTH_AMERICA),
    ]

    for continent, folder_id in continents:
        if not folder_id:
            print(f"No folder ID found for {continent}")
            continue

        print(f"Checking Google Drive folder: {continent} (ID: {folder_id})")

        query = f"'{folder_id}' in parents"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])

        if not files:
            print(f"No images found in {continent} folder on Google Drive!")

        for file in files:
            file_id = file["id"]
            file_name = file["name"]
            print(f"Found image: {file_name} (ID: {file_id})")

            # Google Drive에서 이미지 다운로드
            local_path = download_image_from_drive(file_id, file_name, download_dir)

            # 로컬 경로를 함께 저장 (분석을 위해)
            images.append((continent, local_path, file_id))

    return images

# Google Drive에서 폴더 생성 함수
def create_folder(folder_name, parent_folder_id):
    """Google Drive에 새 폴더 생성 (이미 존재하면 기존 폴더 ID 반환)"""
    query = f"'{parent_folder_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
    response = service.files().list(q=query, fields="files(id)").execute()
    folders = response.get("files", [])

    if folders:
        return folders[0]["id"]  # 기존 폴더 ID 반환

    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id]
    }
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    return folder["id"]


# 분석 완료 후 AFTER 디렉토리에 순서대로 저장
def move_image_to_after(continent, file_id, file_name, analyzed_images):
    after_folder = getattr(settings, f"AFTER_{continent.upper().replace(' ', '_')}", None)
    before_folder = getattr(settings, f"BEFORE_{continent.upper().replace(' ', '_')}", None)

    if not after_folder:
        print(f"No AFTER folder set for {continent}")
        return

    # AFTER 폴더 내 순서대로 저장할 디렉토리 생성
    query = f"'{after_folder}' in parents"
    results = service.files().list(q=query, fields="files(name, id)").execute()
    existing_dirs = sorted([f["name"] for f in results.get("files", []) if f["name"].isdigit()])

    new_index = "000001" if not existing_dirs else f"{int(existing_dirs[-1]) + 1:06d}"
    target_dir = create_folder(new_index, after_folder)

    if before_folder:
        try:
            file_metadata = {
                "addParents": [after_folder],
                "removeParents": [before_folder]
            }
            service.files().update(fileId=file_id, body=file_metadata).execute()
            print(f"Moved {file_name} to {continent}'s AFTER folder.")

        except Exception as e:
            print(f"⚠️ Failed to move {file_name}: {e}")

    # 분석된 결과 이미지 업로드
    for image_path in analyzed_images:
        try:
            file_metadata = {"name": os.path.basename(image_path), "parents": [target_dir]}
            media = MediaFileUpload(image_path, mimetype="image/jpeg")
            service.files().create(body=file_metadata, media_body=media).execute()
            print(f"Uploaded {os.path.basename(image_path)} to {continent}'s AFTER folder.")
        except Exception as e:
            print(f"Failed to upload {os.path.basename(image_path)}: {e}")

    print(f"Uploaded all analyzed images to {continent}'s AFTER folder.")


# BEFORE 디렉토리 초기화
def clear_before_dir():
    """BEFORE 디렉토리 초기화 (모든 파일 삭제)"""
    before_folders = [
        settings.BEFORE_AFRICA,
        settings.BEFORE_ANTARCTICA,
        settings.BEFORE_ASIA,
        settings.BEFORE_EUROPE,
        settings.BEFORE_NORTH_AMERICA,
        settings.BEFORE_OCEANIA,
        settings.BEFORE_SOUTH_AMERICA,
    ]

    for folder_id in before_folders:
        if not folder_id:
            continue

        try:
            query = f"'{folder_id}' in parents"
            results = service.files().list(q=query, fields="files(id)").execute()
            files = results.get("files", [])

            for file in files:
                try:
                    service.files().update(fileId=file["id"], body={"trashed": True}).execute()
                    print(f"Moved {file['name']} to trash.")
                except Exception as e:
                    print(f"Cannot delete {file['id']}: {e}")

        except Exception as e:
            print(f"Error clearing BEFORE directory {folder_id}: {e}")

    print("Cleared BEFORE directory on Google Drive.")