import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pickle

# Memuat model dan encoder
with open('alas_nganjuk.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encode.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Tampilan utama aplikasi
st.title("Aplikasi Random Forest untuk Prediksi Buah")

# Input untuk fitur prediksi
diameter = st.number_input("Diameter (cm)", min_value=0.0, step=0.1)
weight = st.number_input("Berat (gram)", min_value=0.0, step=0.1)
red = st.number_input("Merah (red)", min_value=0.0, step=0.1)
green = st.number_input("Hijau (green)", min_value=0.0, step=0.1)
blue = st.number_input("Biru (blue)", min_value=0.0, step=0.1)

# Tombol prediksi
if st.button("Prediksi"):
    # Pastikan bahwa semua input diisi
    if all(value > 0 for value in [diameter, weight, red, green, blue]):
        # Membuat array input
        features = [[diameter, weight, red, green, blue]]
        
        # Melakukan prediksi
        prediction_encoded = model.predict(features)[0]
        
        # Mendekode hasil prediksi
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        
        st.success(f'Prediksi Buah: {prediction_label}')
    else:
        st.warning("Mohon isi semua input dengan nilai lebih dari 0 untuk melakukan prediksi.")
