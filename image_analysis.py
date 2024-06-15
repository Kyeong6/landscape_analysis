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
