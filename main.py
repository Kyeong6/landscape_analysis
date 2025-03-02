import os
import shutil
import pandas as pd
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

    # 실행 전 디렉토리 데이터 초기화
    def clear_directory(self):
        for directory in [self.images_dir, self.results_dir, self.analyzed_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

    # 크롤링 수행
    def crawl_images(self):
        queries = {
            "africa": "africa landscape real pictures",
            "antarctica": "antarctica landscape real pictures",
            "asia": "asia landscape real pictures",
            "europe": "europe landscape real pictures",
            "north_america": "north america landscape real pictures",
            "oceania": "oceania landscape real pictures",
            "south_america": "south america landscape real pictures"
        }

        for continent, query in queries.items():
            print(f"Crawling for {continent}...")
            self.image_crawler.fetch_images(query, continent)

    # 크롤링된 이미지 가져오기
    def get_local_images(self):
        images = []
        if not os.path.exists(self.images_dir):
            print(f"{self.images_dir} 디렉토리가 존재하지 않음")
            return images
        
        for continent in os.listdir(self.images_dir):
            continent_dir = os.path.join(self.images_dir, continent)
            if os.path.isdir(continent_dir):
                for filename in os.listdir(continent_dir):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_path = os.path.join(continent_dir, filename)
                        images.append((continent, image_path))
        return images
    

    # 이미지 분석 실행
    def analyze_images(self):
        images = self.get_local_images()
        if not images:
            print("분석할 이미지 없음")
            return
        
        analysis_rows = []
        continent_results = {}

        for continent, image_path in images:
            print(f"Analyzing: {image_path}")

            # 분석 실행
            results = self.image_analyzer.run_analysis(image_path, continent)
            if results:
                image_name = os.path.basename(image_path)
                fractal_dimension = results.get("fractal_dimension", "N/A")
                roughness_mean = results.get("surface_roughness_mean", "N/A")
                roughness_std = results.get("surface_roughness_std", "N/A")
                kmeans = results.get("kmeans", {})
                avg_weighted_distance = kmeans.get("avg_weighted_distance", "N/A")
                cluster_ratios = kmeans.get("cluster_ratios", ["N/A"] * 5)
                cluster_ratios = cluster_ratios[:5] + ["N/A"] * (5 - len(cluster_ratios))

                row = [
                    continent, image_name, fractal_dimension, roughness_mean,
                    roughness_std, avg_weighted_distance, *cluster_ratios
                ]
                analysis_rows.append(row)

                if continent not in continent_results:
                    continent_results[continent] = []
                continent_results[continent].append(row)

        for continent, rows in continent_results.items():
            csv_filename = os.path.join(self.results_dir, f"{continent}.csv")
            self.save_results(rows, csv_filename)

        # 분석 결과 저장
        self.save_synthesis_results(analysis_rows)

        print(f"분석 완료 /  총 분석 이미지 수: {len(analysis_rows)}")

    # 분석 결과 저장
    def save_results(self, results, csv_filename):
        headers = [
            "Continent", "Image Name", "Fractal Dimension", "Surface Roughness Mean",
            "Surface Roughness Std", "KMeans Avg Weighted Distance",
            "Cluster 1 Ratio", "Cluster 2 Ratio", "Cluster 3 Ratio", "Cluster 4 Ratio", "Cluster 5 Ratio"
        ]

        df = pd.DataFrame(results, columns=headers)

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
        
        # 대륙별 정렬
        analysis_rows_sorted = sorted(
            analysis_rows,
            key=lambda row: continent_order.index(row[0].lower()) if row[0].lower() in continent_order else 999
        )

        # synthesis.csv 저장
        synthesis_csv = os.path.join(self.results_dir, "synthesis.csv")
        self.save_results(analysis_rows_sorted, synthesis_csv)

    # 전체 실행 프로세스
    def run_process(self):
        self.clear_directory()
        self.crawl_images()
        self.analyze_images()


if __name__ == "__main__":
    processor = ImageProcessor(image_count=1)
    processor.run_process()