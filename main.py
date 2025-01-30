import os
import uvicorn
from fastapi import FastAPI
from drive import get_images_from_drive, move_image_to_after, clear_before_dir
from image_analysis import analyze_image
from spread import save_results_to_sheet

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

    for continent, image_path in images:
        print(f"Analyzing: {image_path}")
        results, analyzed_image_path = analyze_image(image_path, "after")

        if results:
            save_results_to_sheet(continent, os.path.basename(image_path), results)
            move_image_to_after(continent, image_path, analyzed_image_path)
            analysis_results.append({"continent": continent, "image": os.path.basename(image_path)})
            analyzed_count += 1

    clear_before_dir()
    return {"status": "Analysis complete", "total_analyzed": analyzed_count, "details": analysis_results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)