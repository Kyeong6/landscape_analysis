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
