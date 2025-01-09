import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input  # Sesuaikan jika model Anda berbeda
from PIL import Image
import pandas as pd  # Untuk visualisasi tambahan
import gdown
import os
from io import BytesIO
import openpyxl
from openpyxl.chart import BarChart, Reference

# ==========================================
# 1. Load Model (dengan caching agar tidak re-load setiap ada interaksi)
# ==========================================
@st.cache_resource
def load_lung_cancer_model():
    # ID file Google Drive untuk model
    file_id = "1fJkTYJl1k2Eh29bNnQmJRABLf03uEszK"  # Ganti dengan ID file model Anda di Google Drive
    model_path = "lung_cancer_model.h5"
    
    # Cek apakah model sudah ada di lokal, jika tidak, unduh dari Google Drive
    if not os.path.exists(model_path):
        with st.spinner("Mengunduh model dari Google Drive..."):
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
                st.success("Model berhasil diunduh.")
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                return None

    # Memuat model menggunakan TensorFlow
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
        img_height, img_width = 224, 224

        # Konversi ke RGB untuk memastikan 3 channel
        img_pil = img_pil.convert("RGB")

        # Resize gambar menggunakan PIL
        img_pil = img_pil.resize((img_height, img_width))

        # Konversi PIL Image menjadi array
        img_array = image.img_to_array(img_pil)

        # Menambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocessing tambahan
        img_array = preprocess_input(img_array)

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
st.title("Lung Cancer Detection App")
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
# 4.4 Tampilkan Gambar, Prediksi, dan Fitur Tambahan
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
                # Visualisasi Probabilitas
                # -------------------------------------------
                st.write("Probabilitas untuk setiap kelas:")
                prob_values = preds[0] * 100
                prob_labels = [class_labels[k] for k in range(len(class_labels))]

                df_probs = pd.DataFrame({
                    'Kelas': prob_labels,
                    'Probabilitas (%)': prob_values
                }).set_index('Kelas')

                st.bar_chart(df_probs)

                # -------------------------------------------
                # Opsi Download Hasil Prediksi dan Visualisasi (Excel)
                # -------------------------------------------
                st.write("Anda dapat mendownload hasil prediksi dan probabilitas dalam bentuk file Excel dengan visualisasi.")

                # Membuat workbook Excel
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.title = "Hasil Prediksi"

                # Menambahkan hasil prediksi ke Excel
                ws.append(["Label", "Probabilitas (%)"])
                ws.append([predicted_label, probability])
                ws.append([])
                ws.append(["Kelas", "Probabilitas (%)"])
                for index, row in df_probs.reset_index().iterrows():
                    ws.append(row.tolist())

                # Menambahkan grafik batang ke Excel
                chart = BarChart()
                chart.title = "Visualisasi Probabilitas"
                chart.x_axis.title = "Kelas"
                chart.y_axis.title = "Probabilitas (%)"
                data = Reference(ws, min_col=2, min_row=5, max_row=4 + len(prob_labels))
                categories = Reference(ws, min_col=1, min_row=5, max_row=4 + len(prob_labels))
                chart.add_data(data, titles_from_data=False)
                chart.set_categories(categories)
                ws.add_chart(chart, "E8")  # Menempatkan grafik di sel E8

                # Simpan Excel ke buffer
                excel_buffer = BytesIO()
                wb.save(excel_buffer)
                excel_buffer.seek(0)

                # Tombol download untuk Excel
                st.download_button(
                    label="Download Hasil Prediksi dan Visualisasi (Excel)",
                    data=excel_buffer,
                    file_name='hasil_prediksi_visualisasi.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
