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

# Sheet 생성 유무 확인
for continent in continents:
    if continent not in existing_sheets:
        sh.add_worksheet(title=continent, rows="2000", cols="10")

# 결과 저장 함수
def save_results_to_sheet(continent, image_name, analysis_results):
    ws = sh.worksheet(continent)
    new_data = [image_name, json.dumps(analysis_results)]
    ws.append_row(new_data)
    print(f"Data saved to {continent} sheet")