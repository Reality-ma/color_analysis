import numpy as np
from sklearn.cluster import KMeans
from skimage import color, filters, measure, morphology

def color_clustering(image, n_colors=3):
    """对图像进行颜色聚类"""
    if image.shape[2] == 4:  # RGBA 转 RGB
        image = image[:, :, :3]
    img_lab = color.rgb2lab(image)  # 转 LAB 空间
    img_flat = img_lab.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(img_flat)
    labels = kmeans.labels_.reshape(image.shape[:2])
    clustered_img = np.zeros_like(image)
    for i in range(n_colors):
        clustered_img[labels == i] = np.mean(image[labels == i], axis=0)
    return labels, clustered_img.astype(np.uint8)

def extract_layer_edges(label_img):
    """提取层界面边缘"""
    edges = np.zeros_like(label_img, dtype=np.uint8)
    for i in np.unique(label_img):
        mask = (label_img == i).astype(np.float32)
        edge = filters.sobel(mask)  # Sobel 边缘检测
        edge_bin = (edge > 0.01).astype(np.uint8)*255
        edges = np.maximum(edges, edge_bin)
    return edges

def extract_impurity_contours(label_img, original_img):
    """提取杂质轮廓"""
    contours_img = np.zeros_like(original_img)
    for i in np.unique(label_img):
        mask = (label_img == i).astype(np.uint8)
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=20)
        contour_list = measure.find_contours(mask, 0.5)
        for contour in contour_list:
            coords = np.round(contour).astype(int)
            coords[:,0] = np.clip(coords[:,0], 0, contours_img.shape[0]-1)
            coords[:,1] = np.clip(coords[:,1], 0, contours_img.shape[1]-1)
            contours_img[coords[:,0], coords[:,1]] = [0,255,0]
    return contours_img
