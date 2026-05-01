import streamlit as st
import numpy as np
import cv2
import utils

st.set_page_config(layout="wide")

st.title("🚗 ANPR Preprocessing Pipeline")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:

        st.subheader("📌 Original Image")
        st.image(img, channels="BGR", use_column_width=True)

        if st.button("🚀 Process Image"):

            with st.spinner("Processing..."):

                # ======================
                # STEP 1 - PREPROCESSING
                # ======================
                st.subheader("🧪 Step 1: Preprocessing")

                col1, col2 = st.columns(2)

                gray_img = utils.gray_scale(img)
                blur_img = utils.blur_bilateral(gray_img)

                with col1:
                    st.image(gray_img, caption="Grayscale", use_column_width=True)

                with col2:
                    st.image(blur_img, caption="Bilateral Blur", use_column_width=True)

                # ======================
                # STEP 2 - PLATE DETECTION
                # ======================
                st.subheader("🚘 Step 2: Plate Localization")

                img_detected, plate = utils.localize_plate(blur_img)

                col3, col4 = st.columns(2)

                with col3:
                    st.image(img_detected, caption="Detected Plate Area", use_column_width=True)

                with col4:
                    st.image(plate, caption="Cropped Plate", use_column_width=True)

                # ======================
                # STEP 3 - THRESHOLD + MORPH
                # ======================
                st.subheader("⚙️ Step 3: Enhancement")

                plate_thresh = utils.thresholding(plate)
                plate_morph = utils.morphological_operations(plate_thresh)

                col5, col6 = st.columns(2)

                with col5:
                    st.image(plate_thresh, caption="Thresholding", use_column_width=True)

                with col6:
                    st.image(plate_morph, caption="Morphology", use_column_width=True)

                # ======================
                # STEP 4 - CHARACTER DETECTION
                # ======================
                st.subheader("🔍 Step 4: Character Detection")

                contours = utils.find_contours(plate_morph)
                char_candidates = utils.find_characters(contours, plate_morph)
                char_detected = utils.detected_characters(char_candidates, plate_morph)

                st.image(char_detected, caption="Detected Characters", use_column_width=True)

                # ======================
                # STEP 5 - CHARACTER EXTRACTION
                # ======================
                st.subheader("🔡 Step 5: Character Extraction")

                extracted_char = utils.extract_characters_with_pd(char_candidates, plate_morph)
                char_resize = utils.resize_characters(extracted_char)
                char_final = utils.visualize_characters(char_resize)

                cols = st.columns(len(char_final))

                for i, char in enumerate(char_final):
                    with cols[i]:
                        st.image(char, caption=f"Char {i}", clamp=True)

    else:
        st.error("Gagal membaca gambar. Coba format lain.")