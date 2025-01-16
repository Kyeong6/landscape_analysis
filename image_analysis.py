from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import arange, log2, polyfit, polyval
from skimage import color
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from cuml.cluster import KMeans as cuKMeans
import cupy as cp

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


# # k-means 클러스터링 분석
# def kmeans_color_quantization(img_data, k=5):
#     reshaped_data = img_data.reshape((-1, 3))
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(reshaped_data)
#     centers = kmeans.cluster_centers_

#     silhouette_avg = silhouette_score(reshaped_data, labels)
#     print(f"Silhouette Score: {silhouette_avg:.2f}")

#     plt.figure(figsize=(8, 8))
#     plt.pie(
#         np.bincount(labels) / len(labels),
#         labels=[f"Cluster {i+1}" for i in range(k)],
#         colors=[centers[i] / 255 for i in range(k)],
#         startangle=90,
#         counterclock=False,
#     )
#     plt.title("Color Distribution in Image")
#     plt.show()

#     return centers, silhouette_avg

def kmeans_color_quantization(img_data, k=5):
    reshaped_data = img_data.reshape((-1, 3))

    # 데이터를 GPU로 전송
    reshaped_data_gpu = cp.array(reshaped_data)

    # GPU 기반 K-means 클러스터링
    kmeans = cuKMeans(n_clusters=k, random_state=42)
    kmeans.fit(reshaped_data_gpu)

    labels = kmeans.labels_.get()  # GPU에서 CPU로 데이터 복사
    centers = kmeans.cluster_centers_.get()

    # Silhouette Score 계산 (GPU에서는 직접 계산 불가, CPU로 변환 필요)
    silhouette_avg = silhouette_score(reshaped_data, labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")

    plt.figure(figsize=(8, 8))
    plt.pie(
        np.bincount(labels) / len(labels),
        labels=[f"Cluster {i+1}" for i in range(k)],
        colors=[centers[i] / 255 for i in range(k)],
        startangle=90,
        counterclock=False,
    )
    plt.title("Color Distribution in Image (GPU Accelerated)")
    plt.show()

    return centers, silhouette_avg


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
def golden_ratio_analysis(img_data):
    def mutual_information(image, mask_flat):
        # 이미지 평탄화 (1차원으로 변환)
        image_flat = image.flatten()
        mi = -np.sum(image_flat * np.log2(image_flat + 1e-10))  # 엔트로피 기반 계산
        return mi

    def find_best_split(image, direction='horizontal'):
        h, w = image.shape
        max_mi = -np.inf
        best_split = None

        if direction == 'horizontal':
            for i in range(1, h):
                mask = np.zeros((h, w), dtype=int)
                mask[:i, :] = 1
                mask_flat = mask.flatten()
                mi = mutual_information(image, mask_flat)
                if mi > max_mi:
                    max_mi = mi
                    best_split = i
        elif direction == 'vertical':
            for i in range(1, w):
                mask = np.zeros((h, w), dtype=int)
                mask[:, :i] = 1
                mask_flat = mask.flatten()
                mi = mutual_information(image, mask_flat)
                if mi > max_mi:
                    max_mi = mi
                    best_split = i

        return best_split, max_mi

    gray_image = color.rgb2gray(img_data)

    # 수평 및 수직 분할
    h_split, h_mi = find_best_split(gray_image, direction='horizontal')
    v_split, v_mi = find_best_split(gray_image, direction='vertical')

    # 비율 계산
    h_ratio = h_split / gray_image.shape[0] if h_split else None
    v_ratio = v_split / gray_image.shape[1] if v_split else None

    # 황금비율 비교
    golden_ratio = 0.618
    h_difference = abs(h_ratio - golden_ratio) if h_ratio else None
    v_difference = abs(v_ratio - golden_ratio) if v_ratio else None

    print(f"Horizontal split at {h_split} ({h_ratio:.3f}), Difference from golden ratio: {h_difference:.3f}")
    print(f"Vertical split at {v_split} ({v_ratio:.3f}), Difference from golden ratio: {v_difference:.3f}")

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(gray_image, cmap='gray')
    if h_split:
        plt.axhline(h_split, color='red', linestyle='--', label=f'Horizontal: {h_ratio:.3f}')
    if v_split:
        plt.axvline(v_split, color='blue', linestyle='--', label=f'Vertical: {v_ratio:.3f}')
    plt.legend()
    plt.title("Golden Ratio Analysis")
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