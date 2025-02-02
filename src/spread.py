import gspread
import json

# Sheet 연결
gc = gspread.service_account()
sh = gc.open("analysis_result")

# 대륙별 Sheet 리스트 생성
continents = ["Africa", "Antarctica", "Asia", "Europe", "North_america",
              "Oceania", "South_america"]

# 현재 시트 이름 리스트 가져오기
existing_sheets = [worksheet.title for worksheet in sh.worksheets()]

# 컬럼 헤더 정의
COLUMN_HEADERS = [
    "Image Name", "Fractal Dimension", "Surface Roughness Mean", 
    "Surface Roughness Std", "KMeans Avg Weighted Distance",
    "Cluster 1 Ratio", "Cluster 2 Ratio", "Cluster 3 Ratio", "Cluster 4 Ratio", "Cluster 5 Ratio"
]

# 시트가 없으면 생성 & 컬럼 헤더 추가
for continent in continents:
    if continent not in existing_sheets:
        worksheet = sh.add_worksheet(title=continent, rows="2000", cols=str(len(COLUMN_HEADERS)))
        # 컬럼 헤더 추가
        worksheet.append_row(COLUMN_HEADERS)
        print(f"Created sheet '{continent}' with column headers.")

def ensure_headers(ws):
    first_row = ws.row_values(1)
    if not first_row or first_row[0] != "Image Name":  # 0열이 없거나 컬럼명이 다르면 헤더 추가
        ws.insert_row(COLUMN_HEADERS, 1)
        print(f"✅ Updated column headers in '{ws.title}' sheet.")

def format_value(value):
    if isinstance(value, (int, float)):
        return round(value, 4)
    return value

def save_results_to_sheet(continent, image_name, analysis_results):
    """분석 결과를 Google Sheets에 개별 열로 저장"""

    ws = sh.worksheet(continent)
    ensure_headers(ws)

    # 분석 결과 값 확인용 로그 추가
    print("Google Spreadsheet 저장 데이터:")
    print(json.dumps(analysis_results, indent=4))

    if not analysis_results:
        print(f"⚠️ 분석 결과가 없습니다. {continent} sheet에 저장하지 않음.")
        return

    # 개별 수치값 추출
    fractal_dimension = format_value(analysis_results.get("fractal_dimension", "N/A"))
    roughness_mean = format_value(analysis_results.get("surface_roughness_mean", "N/A"))
    roughness_std = format_value(analysis_results.get("surface_roughness_std", "N/A"))
    kmeans_results = analysis_results.get("kmeans", {})
    avg_weighted_distance = format_value(kmeans_results.get("avg_weighted_distance", "N/A"))

    # 클러스터 비율 저장 (최대 5개)
    cluster_ratios = kmeans_results.get("cluster_ratios", ["N/A"] * 5)
    cluster_ratios = [format_value(r) for r in cluster_ratios[:5]] + ["N/A"] * (5 - len(cluster_ratios))

    # 분석 결과를 개별 열에 저장
    row_data = [
        image_name, fractal_dimension, roughness_mean, roughness_std, avg_weighted_distance,
        *cluster_ratios  # 클러스터 비율 5개
    ]

    # Google Sheets에 데이터 추가
    ws.append_row(row_data)
    print(f"Data saved to {continent} sheet")