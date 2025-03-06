import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import arange, log2, polyfit, polyval
from skimage import color
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

# 이미지 분석 객체 정의
class ImageAnalyzer:
    def __init__(self, base_output_dir="analyzed"):
        self.base_output_dir = base_output_dir
        os.makedirs(self.base_output_dir, exist_ok=True)

    # 이미지 로드 및 RGB 변환
    def load_image(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_data = np.array(img)
            return img_data
        except Exception as e:
            print(f"이미지 업로드 실패: {e}")
            return None
        
    # 분석 결과 저장 디렉토리 생성 및 자동 번호 부여
    def setup_output_directory(self, continent):
        continent_dir = os.path.join(self.base_output_dir, continent)
        os.makedirs(continent_dir, exist_ok=True)

        existing_dirs = sorted([d for d in os.listdir(continent_dir) if d.isdigit()])

        new_index = "0001" if not existing_dirs else f"{int(existing_dirs[-1]) + 1:04d}"
        output_path = os.path.join(continent_dir, new_index)
        os.makedirs(output_path, exist_ok=True)

        return output_path
    
    # 이미지 색상 분석
    def compute_color_distribution(self ,img_data, output_path):
        unique_colors, counts = np.unique(
            img_data.reshape(-1, img_data.shape[2]), axis=0, return_counts=True
        )
        sorted_indices = np.argsort(-counts)
        unique_colors = unique_colors[sorted_indices]
        counts = counts[sorted_indices]

        plt.figure(figsize=(20, 10))
        plt.bar(
            range(min(100, len(unique_colors))),
            counts[:100],
            color=[(uc[0] / 255, uc[1] / 255, uc[2] / 255) for uc in unique_colors[:100]],
        )
        plt.title("Top Colors in Image")
        color_path = os.path.join(output_path, "color_spectroscopy.png")
        plt.savefig(color_path)
        plt.close()

        return color_path
    
    # 이미지 거칠기 분석
    def compute_surface_roughness(self, img_data, output_path):
        gray_image = color.rgb2gray(img_data)
        dx = ndimage.sobel(gray_image, 0)
        dy = ndimage.sobel(gray_image, 1)
        magnitude = np.hypot(dx, dy)

        mean_roughness = np.mean(magnitude)
        std_roughness = np.std(magnitude)

        plt.figure(figsize=(8, 6))
        plt.imshow(magnitude, cmap="gray")
        plt.colorbar()
        plt.title("Surface Roughness")
        plt.axis("off")
        surface_path = os.path.join(output_path, "surface_roughness.png")
        plt.savefig(surface_path)
        plt.close()

        return mean_roughness, std_roughness, surface_path
    
    # 프랙탈 차원 분석
    def compute_fractal_dimension(self, img_data, output_path):
        gray_image = color.rgb2gray(img_data)
        sizes = 2 ** arange(8)
        counts = []

        for size in sizes:
            reduced_image = gray_image[::size, ::size]
            counts.append(np.sum(reduced_image > 0))

        logs = log2(sizes)
        log_counts = log2(counts)
        coeffs = polyfit(logs, log_counts, 1)
        fractal_dimension = coeffs[0]

        plt.figure()
        plt.plot(logs, log_counts, "o", label=f"Fractal Dimension = {fractal_dimension:.2f}")
        plt.plot(logs, polyval(coeffs, logs), label="Linear fit")
        plt.xlabel("Log(Box Size)")
        plt.ylabel("Log(Count)")
        plt.legend()
        plt.title("Box Counting Dimension")
        fractal_path = os.path.join(output_path, "fractal_dimension.png")
        plt.savefig(fractal_path)
        plt.close()

        return fractal_dimension, fractal_path
    
    # 이미지 군집화 분석
    def compute_kmeans_clustering(self, img_data, output_path, k=5):
        reshaped_data = img_data.reshape((-1, 3))

        try:
            print(f"[DEBUG] K-Means 실행 중... (데이터 개수: {reshaped_data.shape[0]})")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(reshaped_data)

            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            print(f"[DEBUG] K-Means 클러스터 개수: {len(centers)}")

            # RGB → LAB 변환
            centers_lab = rgb2lab(centers / 255.0)  

            # 클러스터 간 거리 계산 (LAB 색 공간)
            distances = np.sqrt(np.sum((centers_lab[:, np.newaxis, :] - centers_lab[np.newaxis, :, :]) ** 2, axis=2))

            # 중복 없이 클러스터 간 거리만 선택
            # 상삼각행렬 인덱스 (중복 제거)
            triu_indices = np.triu_indices(k, k=1)
            # 상삼각행렬에 해당하는 거리값만 선택
            pairwise_distances = distances[triu_indices]

            # 클러스터 거리 평균 계산
            if len(pairwise_distances) > 0:
                # 100으로 정규화 (0~1 범위)
                contrast_score = round(np.mean(pairwise_distances) / 100, 4)
            else:
                # 클러스터 간 거리 없음
                contrast_score = 0  

            print(f"[DEBUG] 색상 대비 점수 (contrast_score): {contrast_score}")

            # 클러스터 비율 계산
            cluster_ratios = np.bincount(labels, minlength=k) / len(labels)

            # 파이 차트 저장
            plt.figure(figsize=(8, 8))
            plt.pie(cluster_ratios, labels=[f"Cluster {i+1}" for i in range(k)],
                    colors=[centers[i] / 255 for i in range(k)], startangle=90, counterclock=False)
            plt.title("Color Distribution in Image")
            kmeans_path = os.path.join(output_path, "kmeans_pie_chart.png")
            plt.savefig(kmeans_path)
            plt.close()

            return {
                "clusters": centers.tolist(),
                "cluster_ratios": cluster_ratios.tolist(),
                "contrast_score": contrast_score,
                "image_path": kmeans_path
            }
        except Exception as e:
            print(f"[ERROR] K-Means 실행 중 오류 발생: {e}")
            return {
                "clusters": [],
                "cluster_ratios": [],
                "contrast_score": 0,
                "image_path": "N/A"
            }
    
    # 이미지 분석 실행
    def run_analysis(self, image_path, continent):
        img_data = self.load_image(image_path)
        if img_data is None:
            return None

        output_path = self.setup_output_directory(continent)

        fractal_dimension, fractal_path = self.compute_fractal_dimension(img_data, output_path)
        roughness_mean, roughness_std, surface_path = self.compute_surface_roughness(img_data, output_path)
        color_path = self.compute_color_distribution(img_data, output_path)
        kmeans_results = self.compute_kmeans_clustering(img_data, output_path, k=5)

        original_path = os.path.join(output_path, "original.jpg")
        Image.fromarray(img_data).save(original_path)

        results = {
            "fractal_dimension": fractal_dimension,
            "surface_roughness_mean": roughness_mean,
            "surface_roughness_std": roughness_std,
            "kmeans": kmeans_results,
            "image_paths": {
                "original": original_path,
                "color_spectroscopy": color_path,
                "fractal_dimension": fractal_path,
                "surface_roughness": surface_path,
                "kmeans": kmeans_results["image_path"],
            }
        }

        return results