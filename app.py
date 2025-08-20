import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from skimage import color, filters, morphology, feature, exposure
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance

st.title("颜色聚类 + 层界面 & 杂质识别系统")

# --- 图像预处理 ---
def preprocess_image(img):
    gray = color.rgb2gray(np.array(img))
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.03)
    return enhanced

def detect_edges(img_gray):
    edges = feature.canny(img_gray, sigma=1.0)
    edges_dilated = morphology.dilation(edges, morphology.rectangle(1,25))
    return edges_dilated

def detect_anomalies(img_gray):
    thresh_val = filters.threshold_local(img_gray, block_size=35)
    binary = img_gray < thresh_val
    clean = morphology.opening(binary, morphology.square(3))
    return clean

def overlay_contours(base_img, edges, anomalies, edge_color=(255,0,0), anomaly_color=(0,255,0)):
    overlay = np.array(base_img).copy()
    overlay[edges] = edge_color
    overlay[anomalies] = anomaly_color
    return overlay

# --- RGB -> 名称（偏黑敏感） ---
def rgb_to_name(rgb):
    r,g,b = rgb
    brightness = (r+g+b)/3
    if brightness<15:
        return "纯黑"
    elif brightness<35:
        return "黑色"
    elif brightness<55:
        return "深灰"
    elif brightness<85:
        return "灰色"
    elif brightness<120:
        return "浅灰"
    elif brightness<170:
        return "亮灰"
    else:
        return "白色"

# --- Lab 空间合并近似颜色 ---
def merge_similar_colors(avg_colors, threshold=10):
    lab_colors = color.rgb2lab(np.array(avg_colors).reshape(-1,1,3)).reshape(-1,3)
    n = len(lab_colors)
    merged_idx = [-1]*n
    label_count = 0
    for i in range(n):
        if merged_idx[i]!=-1: continue
        merged_idx[i]=label_count
        for j in range(i+1,n):
            if distance.euclidean(lab_colors[i], lab_colors[j])<threshold:
                merged_idx[j]=label_count
        label_count+=1
    return merged_idx

# --- 颜色聚类 ---
def color_clustering(img, n_clusters=4):
    img_np = np.array(img)
    h,w,c = img_np.shape
    reshaped = img_np.reshape((-1,3))
    kmeans = KMeans(n_clusters=n_clusters,n_init=10,random_state=42)
    labels = kmeans.fit_predict(reshaped)
    clustered_img = labels.reshape((h,w))
    clustered_rgb = np.zeros((h,w,3), dtype=np.uint8)
    avg_colors=[]
    for i in range(n_clusters):
        cluster_pixels = reshaped[labels==i]
        avg_rgb = np.mean(cluster_pixels, axis=0)
        avg_colors.append(avg_rgb)
        mask = labels.reshape((h,w))==i
        clustered_rgb[mask] = avg_rgb.astype(np.uint8)
    merged_idx = merge_similar_colors(avg_colors)
    proportions_dict = {}
    color_info=[]
    for i, idx in enumerate(merged_idx):
        mask = clustered_img==i
        prop = np.sum(mask)/mask.size*100
        name = rgb_to_name(avg_colors[i])
        if name in proportions_dict:
            proportions_dict[name]+=prop
        else:
            proportions_dict[name]=prop
        color_info.append((name, prop, avg_colors[i], i))
    color_info_sorted = sorted(color_info, key=lambda x: proportions_dict[x[0]], reverse=True)
    return clustered_rgb, proportions_dict, color_info_sorted, clustered_img

# --- 在单颜色区域识别层界面和杂质 ---
def process_color_region(img_np, mask, edge_color=(255,0,0), anomaly_color=(0,255,0)):
    masked_img = img_np.copy()
    masked_img[~mask] = 0
    gray = color.rgb2gray(masked_img)
    edges = detect_edges(gray)
    anomalies = detect_anomalies(gray)
    overlay = overlay_contours(masked_img, edges, anomalies, edge_color=edge_color, anomaly_color=anomaly_color)
    return overlay, edges, anomalies

# --- 次要颜色边界高亮 ---
def highlight_minor_regions(img, clustered_img, color_labels, proportions, minor_threshold=5, contour_color=(255,0,255), contour_thickness=2):
    highlighted_img = img.copy()
    for label, idx in color_labels.items():
        mask = clustered_img==idx
        area_ratio = np.sum(mask)/mask.size*100
        if area_ratio<minor_threshold:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(highlighted_img, contours, -1, contour_color, contour_thickness)
    return highlighted_img

# --- 侧边栏 ---
st.sidebar.header("次要颜色边界设置")
minor_threshold = st.sidebar.slider("次要颜色面积阈值 (%)",1,15,5,1)
contour_thickness = st.sidebar.slider("边界线宽 (px)",1,5,2,1)
contour_color_choice = st.sidebar.selectbox("次要颜色边界颜色",["紫色","红色","黄色","蓝色"])
contour_color_dict = {"紫色": (255,0,255), "红色": (255,0,0), "黄色": (255,255,0), "蓝色": (0,0,255)}
contour_color = contour_color_dict[contour_color_choice]

# --- 图像输入 ---
option = st.radio("选择图像输入方式", ["上传图片","使用摄像头"])
img = None
if option=="上传图片":
    uploaded_file = st.file_uploader("选择图片文件", type=["jpg","png","jpeg","tif","tiff"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
elif option=="使用摄像头":
    captured_file = st.camera_input("拍摄照片")
    if captured_file is not None:
        img = Image.open(captured_file).convert("RGB")

# --- 主流程 ---
if img is not None:
    img_np = np.array(img)
    n_clusters = st.slider("选择颜色类别数量",3,6,4,1)
    clustered_result, proportions, color_info, clustered_img = color_clustering(img, n_clusters=n_clusters)

    st.subheader("人工修改颜色名称")
    color_labels = {}
    for name, _, avg_rgb, idx in color_info:
        hex_color = '#%02x%02x%02x' % tuple(avg_rgb.astype(int))
        user_label = st.text_input(f"{name} ({hex_color}) 的名称", value=name)
        color_labels[user_label] = idx

    st.subheader("单颜色区域识别结果")
    cols = st.columns(len(color_labels))
    single_color_results = []
    for i,(label, idx) in enumerate(color_la_
