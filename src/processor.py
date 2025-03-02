import os
import shutil
import time
import pandas as pd
import asyncio
from multiprocessing import Pool, cpu_count
from src.image_analysis import ImageAnalyzer
from src.crawling import ImageCrawler

# 이미지 분석 실행 객체 정의
class ImageProcessor:
    def __init__(self, image_count=3):
        self.image_count = image_count
        self.image_crawler = ImageCrawler(count=self.image_count)
        self.image_analyzer = ImageAnalyzer()
        self.results_dir = "results"
        self.images_dir = "images"
        self.analyzed_dir = "analyzed"
        # 크롤링이 끝난 대륙의 이미지 저장할 큐
        self.queue = asyncio.Queue()

        # 수행 시간 저장
        self.crawling_time = 0
        self.analysis_time = 0
        self.total_time = 0

        # 크롤러 실행을 위한 비동기 세팅
        self.continents = {
            "africa": "africa landscape real pictures",
            "antarctica": "antarctica landscape real pictures",
            "asia": "asia landscape real pictures",
            "europe": "europe landscape real pictures",
            "north_america": "north america landscape real pictures",
            "oceania": "oceania landscape real pictures",
            "south_america": "south america landscape real pictures"
        }

    # 실행 전 디렉토리 데이터 초기화
    def clear_directory(self):
        for directory in [self.images_dir, self.results_dir, self.analyzed_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

    # 비동기 기반 크롤링 정의
    async def crawl_continent(self, continent, query):
        print(f"Crawling started for {continent}...")
        crawler = ImageCrawler(count=self.image_count)
        await asyncio.to_thread(crawler.fetch_images, query, continent)
        
        # 크롤링 완료된 대륙을 큐에 추가
        await self.queue.put(continent)

    # 크롤링 수행
    async def crawl_images(self):
        start_time = time.time()
        tasks = [self.crawl_continent(continent, query) for continent, query in self.continents.items()]
        await asyncio.gather(*tasks)
        end_time = time.time()
        self.crawling_time = end_time - start_time

    # 크롤링된 이미지 가져오기
    def get_local_images(self, selected_continents=None):
        images = []
        if not os.path.exists(self.images_dir):
            print(f"{self.images_dir} 디렉토리가 존재하지 않음")
            return images
        
        for continent in os.listdir(self.images_dir):
            if selected_continents and continent not in selected_continents:
                continue
            continent_dir = os.path.join(self.images_dir, continent)
            if os.path.isdir(continent_dir):
                # 파일명을 숫자 기준으로 정렬
                sorted_files = sorted(
                    os.listdir(continent_dir),
                    key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else x
                )
                for filename in sorted_files:
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_path = os.path.join(continent_dir, filename)
                        images.append((continent, image_path))
        return images
    
    # 멀티 프로세싱 기반 이미지 분석 정의
    def analyze_continent(self, continent):
        images = self.get_local_images([continent])
        if not images:
            print(f"{continent}: 분석할 이미지 없음")
            return []
        
        analysis_rows = []

        for _, image_path in images:
            print(f"Analyzing: {image_path}")
            results = self.image_analyzer.run_analysis(image_path, continent)
            if results:
                image_name = os.path.basename(image_path)
                row = [
                    continent, image_name, 
                    results.get("fractal_dimension", "N/A"),
                    results.get("surface_roughness_mean", "N/A"),
                    results.get("surface_roughness_std", "N/A"),
                    results.get("kmeans", {}).get("avg_weighted_distance", "N/A"),
                    *results.get("kmeans", {}).get("cluster_ratios", ["N/A"] * 5)
                ]
                analysis_rows.append(row)

        if analysis_rows:
            csv_filename = os.path.join(self.results_dir, f"{continent}.csv")
            self.save_results(analysis_rows, csv_filename)

        print(f"Analysis completed for {continent}")
        return analysis_rows

    # 이미지 분석 실행
    def analyze_images(self, selected_continents=None):
        
        start_time = time.time()

        # 큐에서 모든 크롤링 완료된 대륙 가져오기
        continents_to_analyze = []
        while not self.queue.empty():
            continents_to_analyze.append(self.queue.get_nowait())
        
        # 병렬 분석 ㅣㅅㄹ행
        with Pool(processes=min(len(self.continents), cpu_count())) as pool:
            results = pool.map(self.analyze_continent, continents_to_analyze)

        analysis_rows = [row for result in results if result for row in result]
        self.save_synthesis_results(analysis_rows)

        self.analysis_time = time.time() - start_time
        print(f"분석 완료 /  총 분석 이미지 수: {len(analysis_rows)}")

    # 분석 결과 저장
    def save_results(self, results, csv_filename):
        headers = [
            "Continent", "Image Name", "Fractal Dimension", "Surface Roughness Mean",
            "Surface Roughness Std", "KMeans Avg Weighted Distance",
            "Cluster 1 Ratio", "Cluster 2 Ratio", "Cluster 3 Ratio", "Cluster 4 Ratio", "Cluster 5 Ratio"
        ]

        # 이미지명으로 정렬
        results_sorted = sorted(results, key=lambda row: int(os.path.splitext(row[1])[0]) if row[1].split('.')[0].isdigit() else row[1])

        df = pd.DataFrame(results_sorted, columns=headers)

        if not os.path.exists(csv_filename):
            df.to_csv(csv_filename, index=False, mode="w")
        else:
            df.to_csv(csv_filename, index=False, mode="a", header=False)

    #  대륙 별 종합 수치 데이터 결과 저장
    def save_synthesis_results(self, analysis_rows):
        continent_order = [
            "africa", "antarctica", "asia", "europe",
            "north_america", "oceania", "south_america"
        ]
        headers = [
            "Continent", "Image Name", "Fractal Dimension", "Surface Roughness Mean",
            "Surface Roughness Std", "KMeans Avg Weighted Distance",
            "Cluster 1 Ratio", "Cluster 2 Ratio", "Cluster 3 Ratio", "Cluster 4 Ratio", "Cluster 5 Ratio"
        ]

        df = pd.DataFrame(analysis_rows, columns=headers)

        # 대륙별 정렬 및 이미지명 정렬
        df["Continent_Order"] = df["Continent"].apply(lambda x: continent_order.index(x.lower()) if x.lower() in continent_order else 999)

        # 파일명이 비어 있는 경우 예외 처리
        def extract_image_num(filename):
            try:
                return int(os.path.splitext(filename)[0])  # "0001.jpg" -> 1
            except ValueError:
                print(f"Warning: 잘못된 파일명 형식 - {filename}")
                return 9999  # 오류 발생 시 가장 마지막에 정렬

        df["Image_Num"] = df["Image Name"].apply(extract_image_num)

        # 정렬 수행 (대륙 우선 → 이미지 번호 순서)
        df = df.sort_values(by=["Continent_Order", "Image_Num"]).drop(columns=["Continent_Order", "Image_Num"])

        # CSV 저장
        synthesis_csv = os.path.join(self.results_dir, "synthesis.csv")
        df.to_csv(synthesis_csv, index=False)

    # 전체 실행 프로세스
    def run_process(self):
        start_time = time.time()

        self.clear_directory()
        asyncio.run(self.crawl_images())
        self.analyze_images()

        end_time = time.time()
        self.total_time = end_time - start_time

        # 수행 시간 출력
        print(f"[Crawling Time]     {self.crawling_time:.2f} 초")
        print(f"[Analysis Time]     {self.analysis_time:.2f} 초")
        print(f"[Image Process]     {self.total_time:.2f} 초")