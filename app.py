import streamlit as st
import numpy as np
from PIL import Image
from utils import color_clustering, extract_layer_edges, extract_impurity_contours

st.title("层状物颜色与层界面识别系统（无需 OpenCV）")

uploaded_file = st.file_uploader("上传层状物图片", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    st.image(img, caption='原图', use_column_width=True)

    n_colors = st.slider("选择颜色类别数量", min_value=2, max_value=10, value=3)

    labels, clustered_img = color_clustering(img_np, n_colors=n_colors)
    st.image(clustered_img, caption='颜色聚类结果', use_column_width=True)

    layer_edges = extract_layer_edges(labels)
    st.image(layer_edges, caption='层界面边缘', use_column_width=True)

    impurity_contours = extract_impurity_contours(labels, clustered_img)
    st.image(impurity_contours, caption='杂质轮廓', use_column_width=True)

    st.write("每个颜色类别独立展示")
    for i in np.unique(labels):
        mask = (labels == i).astype(np.uint8)*255
        st.image(mask, caption=f'颜色类别 {i}', use_column_width=True)
