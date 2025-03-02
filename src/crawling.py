import os
import time
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 이미지 크롤러 객체 정의
class ImageCrawler:
    def __init__(self, base_dir="images", count=3):
        self.base_dir = base_dir
        self.count = count
        self.driver = None
        os.makedirs(self.base_dir, exist_ok=True)

    # Chrome Driver 실행
    def setup_driver(self):
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Chrome Driver 종료
    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def fetch_images(self, query, continent):
        # 대륙 별 디렉토리 설정
        save_dir = os.path.join(self.base_dir, continent)
        os.makedirs(save_dir, exist_ok=True)

        # Chrome Driver 객체
        self.setup_driver()
        search_url = f"https://www.google.com/search?tbm=isch&q={query}"
        self.driver.get(search_url)

        # 이미지 확보(무한 스크롤)
        max_attempts = 20
        attempts = 0
        while attempts < max_attempts:
            image_elements = self.driver.find_elements(By.CSS_SELECTOR, ".H8Rx8c")
            if len(image_elements) >= self.count * 2:
                break
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            attempts += 1

        download_cnt = 0
        i = 0

        print(f"=== 이미지 수집 시작 ===")

        # 썸네일 클릭 후 원본 이미지 다운로드
        while download_cnt < self.count and i < len(image_elements):
            try:
                # 썸네일 클릭
                self.driver.execute_script("arguments[0].click();", image_elements[i])
                time.sleep(2)

                # 원본 이미지 찾기 (XPath 방식)
                original_img = self.driver.find_element(By.XPATH, "//img[contains(@class, 'FyHeAf')]")
                image_url = original_img.get_attribute("src")

                # 파일 저장
                image_filename = f"{download_cnt+1:04d}.jpg"
                image_path = os.path.join(save_dir, image_filename)
                
                try:
                    urllib.request.urlretrieve(image_url, image_path)
                    print(f"다운로드 완료: {image_path}")
                    download_cnt += 1
                except Exception as e:
                    print(f"{image_url} 다운로드 실패: {e}")

            except Exception as e:
                print(f"썸네일 {i} 클릭 또는 원본 이미지 추출 실패: {e}")

            # 실패시 다음 썸네일로 이동
            i += 1
        
        print('=== 이미지 수집 종료 ===')
        self.driver.quit()