import os
import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model dan class names
@st.cache_resource
def get_model():
    return tf.keras.models.load_model(os.path.join(BASE_DIR, 'model_P2G7_Rezha_Aulia.h5'))

@st.cache_resource
def get_class_names():
    with open(os.path.join(BASE_DIR, 'class_names.json'), 'r') as f:
        return json.load(f)

model = get_model()
class_names = get_class_names()

# Halaman
st.set_page_config(
    page_title='Deteksi Penyakit Daun Jagung',
    page_icon='🌽',
    layout='wide'
)

# Judul
st.title('🌽 Deteksi Penyakit Daun Jagung')
st.markdown('''
Aplikasi ini menggunakan model CNN dengan Transfer Learning EfficientNetB0 untuk medeteksi
untuk mendeteksi penyakit daun jagung dari foto yang diambil.
''')

# Sidebar info
st.sidebar.title('ℹ️ Informasi')
st.sidebar.markdown('''
Kelas yang dapat dideteksi:
🔴 Blight (Hawar Daun)
🟠 Common Rust (Karat Daun)
⚪ Gray Leaf Spot (Bercak Abu-abu)
🟢 Healthy (Sehat)

Model : EfficientNetB0
Accuracy : 94%
''')

# Upload Gambar
st.subheader('Upload Foto Daun Jagung')
uploaded_file = st.file_uploader(
    'Pilih gambar daun Jagung',
    type=['jpg', 'jpeg', 'png']
)
if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Gambar yang diupload')
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True)
    with col2:
        st.subheader('Hasil Prediksi')

        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = img_to_array(img_resized)
        img_preprocessed = preprocess_input(img_array)
        img_expanded = np.expand_dims(img_preprocessed, axis=0)

        # Prediksi
        with st.spinner ('Menganalisis gambar'):
            pred_prob = model.predict(img_expanded, verbose=0)
            pred_idx = np.argmax(pred_prob)
            pred_label = class_names[pred_idx]
            confidence = np.max(pred_prob) * 100

        # Warna berdasarkan kelas
        color_map = {
            'Blight (Hawar Daun)':'🔴',
            'Common Rust (Karat Daun)':'🟠',
            'Gray Leaf Spot (Bercak Abu-abu)':'⚪',
            'Healthy (Sehat)':'🟢'    
        }
        emoji = color_map.get(pred_label, '🥬')

        st.success(f'Hasil: {emoji} {pred_label}')
        st.metric('Confidence', f'{confidence:.1f}%')

        # Probability semua kelas
        st.subheader('Probability per Kelas')
        for i,(cls, prob) in enumerate(zip(class_names, pred_prob[0])):
            st.progress(float(prob), text=f'{cls}: {prob*100:.1f}%')

        # Penjelasan hasil
        st.subheader('Keterangan')
        keterangan = {
            'Blight': 'Hawar Daun, segera lakukan penanganan',
            'Common_Rust': 'Karat Daun, monitor perkembangan ',
            'Gray_Leaf_Spot': 'Bercak abu-abu, kurangi kelembapan dan rotasi tanaman',
            'Healthy': 'Daun Sehat, tidak perlu penanganan khusus'
        }
        st.info(keterangan.get(pred_label, ''))
