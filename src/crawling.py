import os
import time
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def img_crawler(query, count, save_dir):

    # 디렉토리 존재 확인
    os.makedirs(save_dir, exist_ok=True)
    
    # Chrome Webdriver 생성
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # 구글 이미지 검색 결과 페이지 이동
    search_url = "https://www.google.com/search?tbm=isch&q=" + query
    driver.get(search_url)
    

    # 무한 스크롤 처리: count 이상 이미지가 로드될 때까지 스크롤
    max_attempts = 20
    attempts = 0
    while attempts < max_attempts:
        image_elements = driver.find_elements(By.CSS_SELECTOR, ".H8Rx8c")
        if len(image_elements) >= count * 2:
            break
        driver.find_element(By.CSS_SELECTOR, "body").send_keys(Keys.END)
        time.sleep(1) 
        attempts += 1

    # 구글 이미지 결과 페이지 내에서 이미지 정보 포함하는 요소 선택
    image_info_list = driver.find_elements(By.CSS_SELECTOR, ".H8Rx8c")

    print(f"=== 이미지 수집 시작 ===")

    image_and_name_list = []
    download_cnt = 0
    i = 0

    # count만큼 다운로드 될 때까지 반복
    while download_cnt < count and i < len(image_info_list):
        image_info = image_info_list[i]
        # 다음 썸네일로 이동
        i += 1
        try:
            # 썸네일 클릭
            driver.execute_script("arguments[0].click();", image_info)

            # 원본 이미지가 로드될 때까지 잠시 대기
            time.sleep(2)
            
            # XPath 방식으로 원본 이미지 요소 추출
            original_img = driver.find_element(By.XPATH, "//img[contains(@class, 'FyHeAf')]")
            save_image = original_img.get_attribute('src')
            
            # 저장할 파일명 생성
            image_filename = f"{download_cnt+1:04d}.jpg"
            image_filepath = os.path.join(save_dir, image_filename)

            # 원본 이미지 다운로드 시도
            try:
                urllib.request.urlretrieve(save_image, image_filepath)
                print(f"다운로드 완료: {image_filepath}")
                image_and_name_list.append((save_image, image_filepath))
                download_cnt += 1  # 성공 시에만 카운트 증가
            except Exception as e:
                print(f"{save_image} 다운로드 실패: {e}")
                # 실패 시 다음 썸네일 시도
                continue
    
        except Exception as e:
            print(f"썸네일 {i} 클릭 또는 원본 추출 실패: {e}")
            # 다음 썸네일 시도
            continue  

    print('=== 이미지 수집 종료 ===')
    driver.close()