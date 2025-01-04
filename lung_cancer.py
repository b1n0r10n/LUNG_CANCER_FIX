import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input  # Sesuaikan jika model Anda berbeda
from PIL import Image
import os
import pandas as pd  # Untuk visualisasi tambahan

# ==========================================
# 1. Load Model (dengan caching agar tidak re-load setiap ada interaksi)
# ==========================================
@st.cache_resource
def load_lung_cancer_model():
    """
    Fungsi untuk memuat model lung cancer.
    Menggunakan st.cache_resource agar proses load hanya dilakukan sekali.
    """
    model_path = 'lung_cancer_model.h5'  # Pastikan path ini benar
    if not os.path.exists(model_path):
        st.error(f"Gagal memuat model: File tidak ditemukan di path '{model_path}'")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_lung_cancer_model()

# Pastikan model berhasil dimuat sebelum melanjutkan
if model is None:
    st.stop()

# ==========================================
# 2. Dictionary Class Labels
# ==========================================
class_labels = {
    0: 'adenocarcinoma',
    1: 'large.cell.carcinoma',
    2: 'normal',
    3: 'squamous.cell.carcinoma'
}

# ==========================================
# 3. Fungsi Prediksi
# ==========================================
def predict_lung_cancer(img_pil):
    """
    Menerima input berupa PIL Image, melakukan preprocessing,
    lalu mengembalikan label dan probabilitas kelas terprediksi.
    """
    try:
        # Tentukan ukuran gambar sesuai kebutuhan model
        img_height, img_width = 224, 224

        # Pastikan gambar diubah ke RGB untuk memastikan 3 channel
        img_pil = img_pil.convert("RGB")

        # Resize gambar menggunakan PIL
        img_pil = img_pil.resize((img_height, img_width))

        # Konversi PIL Image menjadi array
        img_array = image.img_to_array(img_pil)

        # Menambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocessing tambahan (sesuaikan dengan kebutuhan model Anda)
        img_array = preprocess_input(img_array)

        # Verifikasi bentuk array
        if img_array.shape != (1, img_height, img_width, 3):
            raise ValueError(f"Ukuran input tidak sesuai: {img_array.shape}. Harus (1, {img_height}, {img_width}, 3)")

        # Prediksi
        predictions = model.predict(img_array)

        # Mendapatkan indeks kelas dengan nilai tertinggi
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels.get(predicted_class, "Unknown")

        # Mengambil probabilitas dari kelas yang diprediksi
        probability = predictions[0][predicted_class] * 100

        return predicted_label, probability, predictions

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        return None, None, None

# ==========================================
# 4. Streamlit App dengan Navigasi
# ==========================================

# -------------------------------------------
# 4.1 Tambahkan Navigasi ke Website Utama
# -------------------------------------------
st.sidebar.title("Navigasi")
main_website_url = "https://k11-cnn-detection.vercel.app/"  # Ganti dengan URL website utama Anda
st.sidebar.markdown(f"[🔙 Kembali ke Website Utama]({main_website_url})")

# -------------------------------------------
# 4.2 Judul dan Deskripsi Aplikasi
# -------------------------------------------
st.title("Lung Cancer Prediction App")
st.write("""
Aplikasi ini menggunakan model CNN untuk mendeteksi apakah gambar CT-Scan 
termasuk dalam kategori **Adenocarcinoma**, **Large Cell Carcinoma**, **Normal**, atau **Squamous Cell Carcinoma**.
Silakan upload gambar CT-Scan di bawah ini untuk melakukan deteksi.
""")

# -------------------------------------------
# 4.3 Widget File Uploader
# -------------------------------------------
uploaded_file = st.file_uploader("Upload Gambar CT-Scan", type=["png", "jpg", "jpeg"])

# -------------------------------------------
# 4.4 Tampilkan Gambar dan Tombol Prediksi
# -------------------------------------------
if uploaded_file is not None:
    try:
        # Membaca file sebagai PIL Image
        img_pil = Image.open(uploaded_file)

        # Menampilkan gambar yang di-upload
        st.image(img_pil, caption="Gambar yang di-upload", use_column_width=True)

        # Tombol prediksi
        if st.button("Prediksi"):
            with st.spinner("Memproses..."):
                predicted_label, probability, preds = predict_lung_cancer(img_pil)

            if predicted_label is not None:
                # Menampilkan hasil prediksi
                st.success(f"Hasil Prediksi: **{predicted_label.capitalize()}**")
                st.info(f"Probabilitas: **{probability:.2f}%**")

                # -------------------------------------------
                # 4.5 Menambahkan Visualisasi Probabilitas
                # -------------------------------------------
                st.write("Probabilitas untuk setiap kelas:")
                prob_dict = class_labels
                prob_values = preds[0] * 100
                prob_labels = [class_labels[k] for k in prob_dict.keys()]

                df_probs = pd.DataFrame({
                    'Kelas': prob_labels,
                    'Probabilitas (%)': prob_values
                }).set_index('Kelas')

                st.bar_chart(df_probs)

                # -------------------------------------------
                # 4.6 Menambahkan Opsi Download Hasil Prediksi
                # -------------------------------------------
                df_result = pd.DataFrame({
                    'Label': [predicted_label],
                    'Probabilitas (%)': [probability]
                })

                st.download_button(
                    label="Download Hasil Prediksi",
                    data=df_result.to_csv(index=False),
                    file_name='hasil_prediksi.csv',
                    mime='text/csv',
                )

    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
