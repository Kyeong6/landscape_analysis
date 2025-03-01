import os
import pandas as pd
from src.image_analysis import analyze_image
from src.crawling import img_crawler

def save_results_to_csv(results, csv_filename="analysis_results.csv"):
    headers = [
        "Continent", "Image Name", "Fractal Dimension", "Surface Roughness Mean",
        "Surface Roughness Std", "KMeans Avg Weighted Distance",
        "Cluster 1 Ratio", "Cluster 2 Ratio", "Cluster 3 Ratio", "Cluster 4 Ratio", "Cluster 5 Ratio"
    ]
    
    # 결과 DataFrame 생성
    df = pd.DataFrame(results, columns=headers)
    
    if not os.path.exists(csv_filename):
        df.to_csv(csv_filename, index=False, mode="w")
    else:
        df.to_csv(csv_filename, index=False, mode="a", header=False)

def get_local_images(base_dir="images"):
    images = []
    if not os.path.exists(base_dir):
        print(f"디렉토리 '{base_dir}'가 존재하지 않음")
        return images
    
    for continent in os.listdir(base_dir):
        continent_dir = os.path.join(base_dir, continent)
        if os.path.isdir(continent_dir):
            for filename in os.listdir(continent_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(continent_dir, filename)
                    images.append((continent, image_path))
    return images

def crawl_all_images():
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
        save_dir = os.path.join("images", continent)
        print(f"Crawling for {continent}...")
        img_crawler(query, count=1, save_dir=save_dir)

def main():

    # 이미지 크롤링 실행
    crawl_all_images()

    # images 디렉토리에서 각 대륙별 이미지 목록 가져오기
    images = get_local_images("images")
    if not images:
        print("분석할 이미지 존재하지 않음")
        return
    
    analysis_rows = []
    continent_results = {}
    analyzed_count = 0

    for continent, image_path in images:
        print(f"Analyzing: {image_path}")

        # 대률별 분석 결과 저장 폴더 지정
        output_base = os.path.join("analyzed", continent)

        # 이미지 분석 실행(분석 결과 'analyzed' 디렉토리 저장)
        results = analyze_image(image_path, output_base)
        if results:
            image_name = os.path.basename(image_path)
            fractal_dimension = results.get("fractal_dimension", "N/A")
            roughness_mean = results.get("surface_roughness_mean", "N/A")
            roughness_std = results.get("surface_roughness_std", "N/A")
            kmeans = results.get("kmeans", {})
            avg_weighted_distance = kmeans.get("avg_weighted_distance", "N/A")
            cluster_ratios = kmeans.get("cluster_ratios", ["N/A"] * 5)
            # 클러스터 비율은 최대 5개로 맞추기
            cluster_ratios = cluster_ratios[:5] + ["N/A"] * (5 - len(cluster_ratios))
            row = [
                continent, image_name, fractal_dimension, roughness_mean,
                roughness_std, avg_weighted_distance, *cluster_ratios
            ]
            analysis_rows.append(row)

            if continent not in continent_results:
                continent_results[continent] = []
            continent_results[continent].append(row)
            
            analyzed_count += 1

    # results 디렉토리 생성
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 각 대륙별 csv 파일 저장
    for continent, rows in continent_results.items():
        csv_filename = os.path.join(results_dir, f"{continent}.csv")
        save_results_to_csv(rows, csv_filename)

    # synthesis.csv 대륙별 순서대로 정렬
    continent_order = ["africa", "antarctica", "asia", "europe", "north_america", "oceania", "south_america"]
    analysis_rows_sorted = sorted(
        analysis_rows,
        key=lambda row: continent_order.index(row[0].lower()) if row[0].lower() in continent_order else 999
    )

    # 전체 결과를 합친 synthesis.csv 파일 저장
    synthesis_csv = os.path.join(results_dir, "synthesis.csv")
    save_results_to_csv(analysis_rows_sorted, synthesis_csv)
    print(f"분석 완료. 총 분석 이미지 수: {analyzed_count}")

if __name__ == "__main__":
    main()