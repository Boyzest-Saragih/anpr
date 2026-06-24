import streamlit as st
import numpy as np
import cv2
import utils

st.set_page_config(layout="wide")

st.title("Visualisasi Pipeline ANPR processing image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:

        st.markdown("### Input")
        st.image(img, channels="BGR", width=300)

        if st.button("Run Pipeline"):

            with st.spinner("Processing..."):

                # ======================
                # STEP 1
                # ======================
                gray = utils.gray_scale(img)
                blur = utils.blur_bilateral(gray)

                # ======================
                # STEP 2
                # ======================
                detected, plate = utils.localize_plate(blur)

                # ======================
                # STEP 3
                # ======================
                thresh = utils.thresholding(plate)
                morph = utils.morphological_operations(thresh)

                # ======================
                # STEP 4
                # ======================
                contours = utils.find_contours(morph)
                chars = utils.find_characters(contours, morph)
                detected_chars = utils.detected_characters(chars, morph)

                # ======================
                # STEP 5
                # ======================
                extracted = utils.extract_characters_with_pd(chars, morph)
                resized = utils.resize_characters(extracted)
                final_chars = utils.visualize_characters(resized)

                # ======================
                # PIPELINE DISPLAY
                # ======================
                st.markdown("---")
                st.markdown("### Pipeline")

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.caption("Grayscale")
                    st.image(gray, width=250)

                    st.caption("Blur")
                    st.image(blur, width=250)

                with col2:
                    st.caption("Detected Area")
                    st.image(detected, width=250)

                    st.caption("Plate")
                    st.image(plate, width=250)

                with col3:
                    st.caption("Threshold")
                    st.image(thresh, width=250)

                    st.caption("Morphology")
                    st.image(morph, width=250)

                with col4:
                    st.caption("Contours")
                    st.image(detected_chars, width=250)

                with col5:
                    st.caption("Characters")

                    if not final_chars:
                        st.warning("No char")
                    else:
                        for c in final_chars:
                            st.image(c, width=50, clamp=True)

    else:
        st.error("Gagal membaca gambar.")