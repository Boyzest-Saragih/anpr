import streamlit as st
import pandas as pd

print("test")

st.title("ANPR Preprocessing")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    print("File uploaded successfully.")
    st.image(uploaded_file, caption='Original Image', use_column_width=True)
    # tampilkan original
    # tampilkan tiap step

execution_btn = st.button("Process Image")

if execution_btn:
    print("Processing image...")
    # proses image
    st.image(uploaded_file, caption='Processed Image', use_column_width=True)