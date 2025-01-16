from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import arange, log2, polyfit, polyval
from skimage import color
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, rgb2gray

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


# 색상 분석
def chromo_spectroscopy(img_data):
    unique_colors, counts = np.unique(
        img_data.reshape(-1, img_data.shape[2]), axis=0, return_counts=True
    )
    sorted_indices = np.argsort(-counts)
    unique_colors = unique_colors[sorted_indices]
    counts = counts[sorted_indices]

    # 상위 100 색상 시각화
    n = min(100, len(unique_colors))
    plt.figure(figsize=(20, 10))
    plt.bar(
        range(n),
        counts[:n],
        color=[(uc[0] / 255, uc[1] / 255, uc[2] / 255) for uc in unique_colors[:n]],
    )
    plt.title(f"Top {n} Colors in Image")
    plt.xlabel("Color Rank")
    plt.ylabel("Frequency")
    plt.show()

    return pd.DataFrame(
        {"RGB Value": [tuple(uc) for uc in unique_colors], "Frequency": counts}
    )


# 프랙탈 차원 계산
def box_counting_dimension(img_data):
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
    plt.show()

    return fractal_dimension


def kmeans_color_quantization(img_data, k=5):
    reshaped_data = img_data.reshape((-1, 3))

    # CPU 기반 K-means 클러스터링
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reshaped_data)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # LAB 색 공간 변환
    centers_lab = rgb2lab(centers.reshape(1, -1, 3) / 255.0).reshape(-1, 3)

    # 색상 대비 계산 (LAB 거리)
    distances = np.sqrt(np.sum((centers_lab[:, np.newaxis, :] - centers_lab[np.newaxis, :, :]) ** 2, axis=2))

    # 최대 거리로 정규화
    max_distance = np.max(distances)  # 최대 거리
    normalized_distances = distances / max_distance  # 정규화

    # 가중치 기반 색상 대비 계산
    weighted_distances = []
    for i in range(k):
        for j in range(i + 1, k):  # i < j로 제한하여 중복 제거
            spatial_distance = np.linalg.norm(np.array([i, j]))  # 클러스터 위치 간 거리 계산
            weight = 1 / (1 + spatial_distance)  # 가중치 계산
            weighted_contrast = weight * normalized_distances[i, j]  # 정규화된 거리와 가중치 적용
            weighted_distances.append(weighted_contrast)

    # 가중치 평균 계산
    avg_weighted_distance = np.mean(weighted_distances)

    # 출력
    print("Normalized LAB Distance Matrix and Weighted Distances:")
    for i in range(k):
        print(f"Cluster {i+1}: RGB={tuple(map(int, centers[i]))}")
    print("\nWeighted LAB Distances (Normalized):")
    for i in range(k):
        for j in range(i + 1, k):
            spatial_distance = np.linalg.norm(np.array([i, j]))
            weight = 1 / (1 + spatial_distance)
            weighted_contrast = weight * normalized_distances[i, j]
            print(f"  Weighted Distance between Cluster {i+1} and Cluster {j+1}: {weighted_contrast:.5f}")
    print(f"\nAverage Weighted Distance (Normalized): {avg_weighted_distance:.5f}")

    # 파이 차트 시각화
    plt.figure(figsize=(8, 8))
    plt.pie(
        np.bincount(labels) / len(labels),
        labels=[f"Cluster {i+1}" for i in range(k)],
        colors=[centers[i] / 255 for i in range(k)],
        startangle=90,
        counterclock=False,
    )
    plt.title("Color Distribution in Image")
    plt.show()

    return centers, normalized_distances, avg_weighted_distance



# 이미지 거칠기 분석
def surface_roughness(img_data):
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
    plt.show()

    print(f"Mean Surface Roughness: {mean_roughness:.4f}")
    print(f"Standard Deviation of Surface Roughness: {std_roughness:.4f}")


# 황금비율 분석
def mutual_information(image, mask):
    """
    Calculate mutual information between the image and the mask.
    """
    image_flat = image.flatten()
    mask_flat = mask.flatten()
    joint_hist, _, _ = np.histogram2d(image_flat, mask_flat, bins=256, range=[[0, 1], [0, 1]], density=True)
    joint_prob = joint_hist / np.sum(joint_hist)

    marginal_image = np.sum(joint_prob, axis=1)
    marginal_mask = np.sum(joint_prob, axis=0)

    joint_entropy = -np.nansum(joint_prob * np.log2(joint_prob + 1e-10))
    entropy_image = -np.nansum(marginal_image * np.log2(marginal_image + 1e-10))
    entropy_mask = -np.nansum(marginal_mask * np.log2(marginal_mask + 1e-10))

    return entropy_image + entropy_mask - joint_entropy


def find_best_split(image, direction='horizontal', step_size=5):
    """
    Find the best split position based on maximum mutual information gain.
    """
    h, w = image.shape
    max_mi = -np.inf
    best_split = None

    if direction == 'horizontal':
        for i in range(step_size, h - step_size, step_size):
            mask = np.zeros((h, w), dtype=int)
            mask[:i, :] = 1
            mi = mutual_information(image, mask)
            if mi > max_mi:
                max_mi = mi
                best_split = i
    elif direction == 'vertical':
        for i in range(step_size, w - step_size, step_size):
            mask = np.zeros((h, w), dtype=int)
            mask[:, :i] = 1
            mi = mutual_information(image, mask)
            if mi > max_mi:
                max_mi = mi
                best_split = i

    return best_split, max_mi


def golden_ratio_analysis(img_data, step_size=5):
    """
    Perform golden ratio analysis using mutual information.
    """
    # Convert image to grayscale
    gray_image = rgb2gray(img_data)

    # Find optimal horizontal and vertical splits
    h_split, h_mi = find_best_split(gray_image, direction='horizontal', step_size=step_size)
    v_split, v_mi = find_best_split(gray_image, direction='vertical', step_size=step_size)

    # Calculate proportions
    h_ratio = h_split / gray_image.shape[0] if h_split else 0.0
    v_ratio = v_split / gray_image.shape[1] if v_split else 0.0

    # Compare to golden ratio
    golden_ratio = 0.618
    h_difference = abs(h_ratio - golden_ratio)
    v_difference = abs(v_ratio - golden_ratio)

    print(f"Horizontal split at {h_split} ({h_ratio:.3f}), Difference from golden ratio: {h_difference:.3f}")
    print(f"Vertical split at {v_split} ({v_ratio:.3f}), Difference from golden ratio: {v_difference:.3f}")

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(gray_image, cmap='gray')

    if h_split:
        plt.axhline(h_split, color='red', linestyle='--', label=f'Horizontal: {h_ratio:.3f}')
        plt.axhline(golden_ratio * gray_image.shape[0], color='green', linestyle='--', label='Golden Ratio Horizontal')

    if v_split:
        plt.axvline(v_split, color='blue', linestyle='--', label=f'Vertical: {v_ratio:.3f}')
        plt.axvline(golden_ratio * gray_image.shape[1], color='orange', linestyle='--', label='Golden Ratio Vertical')

    plt.legend()
    plt.title("Golden Ratio Analysis with Mutual Information")
    plt.axis("off")
    plt.show()


# 메인 함수
def main():
    image_path = "./image/scotland.jpg"
    img_data = load_image(image_path)

    if img_data is not None:
        print("Performing Chromo-spectroscopy")
        chromo_spectroscopy(img_data)

        print("Calculating Fractal Dimension")
        box_counting_dimension(img_data)

        print("Analyzing Surface Roughness")
        surface_roughness(img_data)

        print("Performing K-means Clustering")
        kmeans_color_quantization(img_data, k=5)

        print("Performing Golden Ratio Analysis")
        golden_ratio_analysis(img_data)


if __name__ == "__main__":
    main()