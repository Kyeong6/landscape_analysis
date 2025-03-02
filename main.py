from src.processor import ImageProcessor

# 이미지 분석 프로세스 실행
if __name__ == "__main__":
    processor = ImageProcessor(image_count=3)
    processor.run_process()