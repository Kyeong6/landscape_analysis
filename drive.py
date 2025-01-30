import os
import shutil
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Google Drive 인증
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# 디렉토리 설정
before_dir = "before"
after_dir = "after"

# 이미지 목록 가져오기
def get_images_from_drive():
    images = []
    for continent in os.listdir(before_dir):
        continent_path = os.path.join(before_dir, continent)
        if os.path.isdir(continent_path):
            for img_file in os.listdir(continent_path):
                images.append((continent, os.path.join(continent_path, img_file)))
    return images

# 분석 완료 후 이동
def move_image_to_after(continent, image_path, analyzed_image_path):
    target_dir = os.path.join(after_dir, continent, os.path.basename(image_path).split('.')[0])
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(image_path, os.path.join(target_dir, "original.jpg"))
    shutil.move(analyzed_image_path, os.path.join(target_dir, "analyzed.jpg"))

# before 디렉토리 초기화
def clear_before_dir():
    for continent in os.listdir(before_dir):
        continent_path = os.path.join(before_dir, continent)
        if os.path.isdir(continent_path):
            shutil.rmtree(continent_path)
    os.makedirs(before_dir, exist_ok=True)