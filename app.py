import streamlit as st
import numpy as np
from PIL import Image
from skimage import color, feature, filters, morphology, exposure
from sklearn.cluster import KMeans
import cv2

st.title("颜色聚类 + 层界面和杂质识别")

# --- 图像输入 ---
uploaded_file = st.file_uploader("上传图片", type=["jpg","png","jpeg","tif","tiff"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    
    # --- 用户设置 ---
    n_clusters = st.slider("选择颜色类别数量", 3, 6, 4)
    
    # --- 颜色聚类 ---
    h,w,c = img_np.shape
    reshaped = img_np.reshape((-1,3))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(reshaped)
    clustered_img = labels.reshape((h,w))
    
    st.subheader("颜色聚类结果")
    for i in range(n_clusters):
        mask = clustered_img==i
        cluster_rgb = np.zeros_like(img_np)
        cluster_rgb[mask] = img_np[mask]
        st.image(cluster_rgb, caption=f"颜色类别 {i+1}", use_container_width=True)
    
    # --- 对每个颜色区域进行层界面和杂质识别 ---
    st.subheader("层界面和杂质识别结果")
    for i in range(n_clusters):
        mask = clustered_img==i
        masked_img = img_np.copy()
        masked_img[~mask] = 0
        gray = color.rgb2gray(masked_img)
        
        # 层界面识别
        edges = feature.canny(gray, sigma=1.0)
        edges_dilated = morphology.dilation(edges, morphology.rectangle(1,25))
        
        # 杂质识别
        thresh_val = filters.threshold_local(gray, block_size=35)
        anomalies = gray < thresh_val
        anomalies = morphology.opening(anomalies, morphology.square(3))
        
        # 叠加轮廓
        overlay = masked_img.copy()
        overlay[edges_dilated] = [255,0,0]       # 界面红色
        overlay[anomalies] = [0,255,0]           # 杂质绿色
        
        st.image(overlay, caption=f"颜色类别 {i+1} 层界面+杂质", use_container_width=True)
