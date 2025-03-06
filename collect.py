import os
import asyncio
from src.crawling import ImageCrawler

# 검색어 기입
QUERY = "landscape images"

# 수집할 이미지 개수
IMAGE_COUNT = 5 

# 이미지 수집 객체 정의
class ImageCollector:
    def __init__(self, query, count):
        self.query = query
        self.count = count
        
        # 저장할 디렉토리
        self.collect_dir = "collect"
        os.makedirs(self.collect_dir, exist_ok=True)

    async def collect_images(self):
        print(f"=== 이미지 수집 시작: {self.query} ({self.count}장) ===")
        
        # ImageCrawler를 사용하여 이미지 수집
        crawler = ImageCrawler(base_dir=self.collect_dir, count=self.count)
        await asyncio.to_thread(crawler.fetch_images, self.query, "")
        
        print(f"=== 이미지 수집 완료: {self.collect_dir} ===")

if __name__ == "__main__":
    collector = ImageCollector(query=QUERY, count=IMAGE_COUNT)
    asyncio.run(collector.collect_images())