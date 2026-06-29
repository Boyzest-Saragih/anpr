import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import torch
import tensorflow as tf

torch.set_num_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import cv2
cv2.setNumThreads(0) 

import streamlit as st
import numpy as np
from src.pipeline import recognize_plate

st.set_page_config(
    page_title="ANPR",
    layout="wide"
)

st.title("🚗 Automatic Number Plate Recognition")

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Original Image",
        use_container_width=True
    )

    if st.button("Detect Plate"):

        with st.spinner("Processing..."):

            result = recognize_plate(image)

        st.success("Detection Finished")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Detected Plate")

            st.image(
                cv2.cvtColor(result["plate_image"], cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with col2:

            st.subheader("Recognition Result")
            st.markdown(f"# {result['plate_number']}")
            st.write(f"**Confidence:** {result['confidence']}%") 

        st.subheader("Detected Characters")

        cols = st.columns(len(result["characters"]))

        for i, char in enumerate(result["characters"]):

            cols[i].image(
                char,
                use_container_width=True,
                clamp=True
            )