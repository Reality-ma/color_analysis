import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import filters, morphology

def color_clustering(image, n_colors=3):
    """对图像进行颜色聚类"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_flat = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(img_flat)
    labels = kmeans.labels_.reshape(img_rgb.shape[:2])
    clustered_img = kmeans.cluster_centers_[labels].astype(np.uint8)
    return labels, clustered_img

def extract_layer_edges(label_img):
    """提取层界面边缘"""
    edges = np.zeros_like(label_img, dtype=np.uint8)
    for i in np.unique(label_img):
        mask = (label_img == i).astype(np.uint8)
        edge = cv2.Canny(mask*255, 50, 150)
        edges = cv2.bitwise_or(edges, edge)
    return edges

def extract_impurity_contours(label_img, original_img):
    """提取颜色对比明显的杂质轮廓"""
    contours_img = np.zeros_like(original_img)
    for i in np.unique(label_img):
        mask = (label_img == i).astype(np.uint8)
        # 杂质检测：高对比部分
        edges = cv2.Canny(mask*255, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, (0,255,0), 1)
    return contours_img
