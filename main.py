import os
import uvicorn
from fastapi import FastAPI
from src.drive import get_images_from_drive, move_image_to_after, clear_before_dir
from src.image_analysis import analyze_image
from src.spread import save_results_to_sheet

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Landscape Image Analysis API is Running!"}

@app.post("/analyze")
async def analyze_images():
    images = get_images_from_drive()
    if not images:
        return {"status": "No images found in Google Drive."}

    analyzed_count = 0
    analysis_results = []

    for continent, local_path, file_id in images:  # 로컬 경로 사용
        print(f"Analyzing: {local_path}")

        # 분석 실행
        results, output_path = analyze_image(local_path, "analyzed")  # 로컬 분석

        if results:
            save_results_to_sheet(continent, os.path.basename(local_path), results)

            # 분석된 이미지 파일 목록 가져오기
            analyzed_image_paths = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".png") or f.endswith(".jpg")]

            move_image_to_after(continent, file_id, os.path.basename(local_path), analyzed_image_paths)
            analysis_results.append({"continent": continent, "image": os.path.basename(local_path)})
            analyzed_count += 1

    clear_before_dir()
    return {"status": "Analysis complete", "total_analyzed": analyzed_count, "details": analysis_results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)