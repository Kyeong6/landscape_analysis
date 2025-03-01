import os
import time
import shutil
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def img_crawler(query, count, save_dir):

    # 디렉토리 초기화
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Chrome Webdriver 생성
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # 구글 이미지 검색 결과 페이지 이동
    search_url = "https://www.google.com/search?tbm=isch&q=" + query
    driver.get(search_url)
    

    # 무한 스크롤 처리: count 이상 이미지가 로드될 때까지 스크롤
    max_attempts = 3  # 최대 스크롤 시도 횟수
    attempts = 0
    while attempts < max_attempts:
        image_elements = driver.find_elements(By.CSS_SELECTOR, ".H8Rx8c")
        if len(image_elements) >= count:
            break
        driver.find_element(By.CSS_SELECTOR, "body").send_keys(Keys.END)
        time.sleep(1)  # 페이지 로딩 대기
        attempts += 1

    # 구글 이미지 결과 페이지 내에서 이미지 정보 포함하는 요소 선택
    image_info_list = driver.find_elements(By.CSS_SELECTOR, ".H8Rx8c")
    image_and_name_list = []

    print(f"=== 이미지 수집 시작 ===")

    download_cnt = 0

    # image_info_list를 순회하여 이미지 URL 추출
    for i, image_info in enumerate(image_info_list):
        if download_cnt == count:
            break
        try:
            # 썸네일 클릭
            driver.execute_script("arguments[0].click();", image_info)

            # 원본 이미지가 로드될 때까지 잠시 대기
            time.sleep(2)
            
            # XPath 방식으로 원본 이미지 요소 추출 (클래스에 'FyHeAf'가 포함된 이미지)
            original_img = driver.find_element(By.XPATH, "//img[contains(@class, 'FyHeAf')]")
            save_image = original_img.get_attribute('src')
        except Exception as e:
            print(f"이미지 {i} 추출 실패: {e}")
            continue
            
        # 저장할 파일명 생성
        image_filename = f"{download_cnt+1:04d}.jpg"
        image_filepath = os.path.join(save_dir, image_filename)
        image_and_name_list.append((save_image, image_filepath))
        download_cnt += 1
    
    # 이미지 적재
    for image_url, image_filepath in image_and_name_list:
        try:
            urllib.request.urlretrieve(image_url, image_filepath)
            print(f"다운로드 완료: {image_filepath}")
        except Exception as e:
            print(f"{image_url} 다운로드 실패: {e}")
            # 실패시 다음 썸네일로 넘어감
            continue

    print('=== 이미지 수집 종료 ===')
    driver.close()