import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from numpy import arange, log2, polyfit, polyval
from skimage import color
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, rgb2gray

# 분석된 결과 저장 경로
RESULT_DIR = "after"

# 이미지 로드 및 초기화
def load_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_data = np.array(img)
        return img_data
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# 분석 결과 저장 디렉토리 생성 및 순서대로 파일명 지정
def get_output_dir(base_path):
    existing_dirs = sorted(os.listdir(base_path))
    if not existing_dirs:
        return os.path.join(base_path, "000001")
    
    last_dir = existing_dirs[-1]
    new_index = int(last_dir) + 1
    return os.path.join(base_path, f"{new_index:06d}")

# 색상 분석
def chromo_spectroscopy(img_data, output_path):
    unique_colors, counts = np.unique(
        img_data.reshape(-1, img_data.shape[2]), axis=0, return_counts=True
    )
    sorted_indices = np.argsort(-counts)
    unique_colors = unique_colors[sorted_indices]
    counts = counts[sorted_indices]

    # 시각화 저장
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
def surface_roughness(img_data, output_path):
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


# 프랙탈 차원 계산
def box_counting_dimension(img_data, output_path):
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


def kmeans_color_quantization(img_data, output_path, k=5):
    # CPU 기반 K-means 클러스터링
    reshaped_data = img_data.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reshaped_data)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # LAB 색 공간 변환 및 색상 대비 계산
    centers_lab = rgb2lab(centers.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    distances = np.sqrt(np.sum((centers_lab[:, np.newaxis, :] - centers_lab[np.newaxis, :, :]) ** 2, axis=2))

    # 최대 거리로 정규화
    max_distance = np.max(distances) if np.max(distances) > 0 else 1
    normalized_distances = np.nan_to_num(distances / max_distance)

    # 가중치 기반 색상 대비 계산
    # 가중 평균 계산
    weighted_distances = [
        (1 / (1 + np.linalg.norm(np.array([i, j])))) * normalized_distances[i, j]
        for i in range(k) for j in range(i + 1, k)
    ]
    avg_weighted_distance = np.mean(weighted_distances) if weighted_distances else 0

    # 파이 차트 시각화
    plt.figure(figsize=(8, 8))
    cluster_counts = np.bincount(labels, minlength=k) / len(labels)
    plt.pie(cluster_counts, labels=[f"Cluster {i+1}" for i in range(k)],
            colors=[centers[i] / 255 for i in range(k)], startangle=90, counterclock=False)
    plt.title("Color Distribution in Image")
    kmeans_path = os.path.join(output_path, "kmeans_pie_chart.png")
    plt.savefig(kmeans_path)
    plt.close()

    return {
        "clusters": centers.tolist(),
        "normalized_distances": normalized_distances.tolist(),
        "avg_weighted_distance": avg_weighted_distance,
        "image_path": kmeans_path
    }

# 분석 실행
def analyze_image(image_path, output_base_path):
    img_data = load_image(image_path)
    if img_data is None:
        return None, None
    
    # 순서대로 저장할 디렉토리 지정
    output_path = get_output_dir(output_base_path)
    os.makedirs(output_path, exist_ok=True)

    # 분석 실행
    color_path = chromo_spectroscopy(img_data, output_path)
    fractal_dimension, fractal_path = box_counting_dimension(img_data, output_path)
    roughness_mean, roughness_std, surface_path = surface_roughness(img_data, output_path)
    kmeans_results = kmeans_color_quantization(img_data, output_path, k=5)

    # 원본 이미지 저장
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
    
    return results, output_path