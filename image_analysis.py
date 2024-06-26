from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from numpy import unique, where, arange, log2, polyfit, polyval
from skimage import io
from skimage import color
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

# 이미지에 존재하는 색 확인
def chromo_spectroscopy(image_path):
    img = Image.open(image_path)
    img_data = np.array(img)
    
    unique_colors, counts = np.unique(img_data.reshape(-1, img_data.shape[2]), axis=0, return_counts=True)
    sorted_indices = np.argsort(-counts)
    unique_colors = unique_colors[sorted_indices]
    counts = counts[sorted_indices]
    
    # 모든 색상의 RGB 값과 빈도 출력
    color_data = {'RGB Value': [tuple(uc) for uc in unique_colors],
                  'Frequency': counts}
    color_table = pd.DataFrame(color_data)
    
    # 모든 색상을 막대 그래프로 출력
    n = min(100, len(unique_colors))  
    plt.figure(figsize=(20, 10))
    bars = plt.bar(range(n), counts[:n], color=[(uc[0]/255, uc[1]/255, uc[2]/255) for uc in unique_colors[:n]])
    plt.title(f'Top {n} Colors in Image')
    plt.xlabel('Color Rank')
    plt.ylabel('Frequency')
    plt.xticks(range(n), range(1, n + 1))
    
    plt.show()
    
    return color_table

# 프랙탁 차원 계산
def box_counting_dimension(image_path):
    image = io.imread(image_path)
    flat_image = image.reshape(-1, image.shape[2])
    unique_colors = unique(flat_image, axis=0)

    # 박스 크기 범위 및 빈 박스 수 계산
    sizes = 2**arange(8)  
    counts = []

    for size in sizes:
        # 색상 개수
        reduced_image = unique_colors // size
        reduced_colors = unique(reduced_image, axis=0)
        counts.append(len(reduced_colors))

    # 프랙탈 차원 
    logs = log2(sizes)
    log_counts = log2(counts)
    coeffs = polyfit(logs, log_counts, 1)
    fractal_dimension = coeffs[0]

    plt.figure()
    plt.plot(logs, log_counts, 'o', label=f'Fractal Dimension = {fractal_dimension:.2f}')
    plt.plot(logs, polyval(coeffs, logs), label='Linear fit')
    plt.xlabel('Log(Box Size)')
    plt.ylabel('Log(Count)')
    plt.legend()
    plt.title('Box Counting Dimension')
    plt.show()

    return fractal_dimension

# 이미지의 거칠기 파악
def surface_roughness(image_path):
    # gray scale 변환
    image = io.imread(image_path, as_gray=True)
    
    # Sobel 필터 : x,y 방향의 gradient 계산
    dx = ndimage.sobel(image, 0)  # x 방향의 derivative
    dy = ndimage.sobel(image, 1)  # y 방향의 derivative
    magnitude = np.hypot(dx, dy)  # gradient magnitude
    
    # 평균 및 표준 편차 
    mean_roughness = np.mean(magnitude)
    std_roughness = np.std(magnitude)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude, cmap='gray')
    plt.colorbar()
    plt.title('Image Surface Roughness')
    plt.axis('off')
    plt.show()
    
    print(f'Mean Surface Roughness: {mean_roughness}')
    print(f'Standard Deviation of Surface Roughness: {std_roughness}')


# k-means clustering
def kmeans_color_quantization(image_path, k=5):

    img = Image.open(image_path)
    img = img.convert('RGB')
    img_data = np.array(img)
    
    # 2차원 배열로 변환 (pixel 수 x 3(RGB))
    img_data_reshaped = img_data.reshape((-1, 3))
    
    # k-means clustering 적용
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img_data_reshaped)
    
    # 중심 색상 계산
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # 비율 계산
    label_counts = np.bincount(labels)
    label_ratios = label_counts / len(labels)
    
    # RGB to CIELAB 변환
    centers_lab = color.rgb2lab(centers.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    
    # 클러스터 중심 간의 거리 계산 (CIELAB)
    distances = np.sqrt(np.sum((centers_lab[:, np.newaxis, :] - centers_lab[np.newaxis, :, :]) ** 2, axis=2))
    
    # 거리 기반 가중치 계산
    weight_matrix = np.exp(-distances / distances.max())
    weights = weight_matrix.sum(axis=1)
    
    # 가중치의 평균 값 계산
    mean_weight = np.mean(weights)

    plt.figure(figsize=(8, 8))
    plt.pie(label_ratios, labels=[f'Cluster {i+1}' for i in range(k)], colors=[centers[i]/255 for i in range(k)], startangle=90, counterclock=False)
    plt.title('Color Distribution in Image')
    plt.show()
    
    return centers, label_ratios, centers_lab, weights, mean_weight

# 색상 양자화
def quantize_image(image_path, bits=3):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    
    factor = 256 // (2**bits)
    quantized_image = (image // factor) * factor
    
    return quantized_image

# 엔트로피 계산
def calculate_entropy(image):
    pixels = image.reshape(-1, image.shape[-1])
    _, counts = np.unique(pixels, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 이미지 평탄화
def mutual_information(image, mask_flat):
    image_flat = image.reshape(-1, image.shape[-1])
    image_flat = image_flat[:, 0]  # 이미지의 첫 번째 채널 사용 (흑백으로 간주)
    mi = mutual_info_score(image_flat, mask_flat)
    return 


# 이미지 분할
def partition_image(image, direction='horizontal'):
    h, w, _ = image.shape
    max_mi = -np.inf
    best_partition = None

    if direction == 'horizontal':
        for i in range(1, h):
            mask = np.zeros((h, w), dtype=int)
            mask[:i, :] = 1
            mask_flat = mask.flatten()
            mi = mutual_information(image, mask_flat)
            if mi > max_mi:
                max_mi = mi
                best_partition = i
    else:
        for i in range(1, w):
            mask = np.zeros((h, w), dtype=int)
            mask[:, :i] = 1
            mask_flat = mask.flatten()
            mi = mutual_information(image, mask_flat)
            if mi > max_mi:
                max_mi = mi
                best_partition = i
                
    return best_partition, max_mi


def segment_image(image_path):
    image = quantize_image(image_path)
    h_partition, h_mi = partition_image(image, 'horizontal')
    v_partition, v_mi = partition_image(image, 'vertical')
    
    if h_mi > v_mi:
        print(f"Best partition: Horizontal at {h_partition}")
    else:
        print(f"Best partition: Vertical at {v_partition}")



# 메인 함수 설정
def main():
    image_path = './landscape.jpg'
    print("Performing Chromo-spectroscopy...")
    chromo_spectroscopy(image_path)

    print("Calculating Fractal Dimension...")
    box_counting_dimension(image_path)

    print("Analyzing Surface Roughness and Brightness Contrast...")
    surface_roughness(image_path)

    print("k-mean analyze")
    centers, label_ratios, centers_lab, weights, mean_weight = kmeans_color_quantization(image_path, 5)
    # 클러스터 중심 색상과 비율, LAB 값, 가중치 출력
    for i, (center, ratio, lab, weight) in enumerate(zip(centers, label_ratios, centers_lab, weights)):
        print(f'Cluster {i+1}: RGB={center}, Ratio={ratio}, LAB={lab}, Weight={weight}')
    # 평균 가중치 출력
    print(f'Mean Weight: {mean_weight}')
    
    # segment_image(image_path)

    
# 실행
if __name__ == '__main__':
     main()