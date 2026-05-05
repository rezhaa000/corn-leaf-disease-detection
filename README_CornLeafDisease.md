# 🌽 Corn Leaf Disease Detection

Sistem deteksi penyakit daun jagung berbasis gambar menggunakan Convolutional Neural Network (CNN), dibangun untuk membantu petani dan penyuluh pertanian mengidentifikasi jenis penyakit tanaman jagung secara mandiri langsung dari foto daun menggunakan smartphone.

---

## 📋 Daftar Isi
- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Struktur Repository](#struktur-repository)
- [Alur Project](#alur-project)
- [Hasil Model](#hasil-model)
- [Deployment](#deployment)
- [Tools & Libraries](#tools--libraries)

---

## 📌 Latar Belakang

Jagung merupakan salah satu komoditas pangan strategis di Indonesia. Penyakit seperti **Blight (Hawar)**, **Common Rust (Karat)**, dan **Gray Leaf Spot (Bercak Abu-abu)** dapat menyebabkan gagal panen jika tidak ditangani sejak dini. Inspeksi manual membutuhkan ahli pertanian di lapangan yang tidak selalu tersedia.

Sistem berbasis CNN ini memungkinkan deteksi penyakit lebih cepat dan mandiri — cukup foto daun jagung, model langsung memberikan hasil klasifikasi beserta tingkat kepercayaannya.

---

## 🗃️ Dataset

- **Sumber:** [Corn / Maize Leaf Disease Dataset — Kaggle](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset/data)
- **Total gambar:** 4.188 gambar
- **Kelas:**

| Kelas | Jumlah | Persentase |
|---|---|---|
| Common Rust | 1.306 | 31.2% |
| Healthy | 1.162 | 27.7% |
| Blight | 1.146 | 27.4% |
| Gray Leaf Spot | 574 | 13.7% |

> ⚠️ Dataset bersifat **imbalanced** — Gray Leaf Spot hanya 13.7%, diatasi dengan class weighting.

---

## 📁 Struktur Repository

```
├── P2G7_Rezha_Aulia.ipynb          # Notebook utama (EDA, training, evaluasi)
├── P2G7_Rezha_Aulia_inference.ipynb # Notebook inference
├── deployment/
│   ├── app.py                       # Streamlit web app
│   ├── requirements.txt
│   └── Dockerfile
├── class_names.json                 # Label kelas model
└── url.text                         # Link model & deployment
```

---

## 🔄 Alur Project

### 1. Exploratory Data Analysis (EDA)
- Analisis distribusi kelas → ditemukan ketidakseimbangan pada Gray Leaf Spot (13.7%)
- Sample gambar per kelas → Gray Leaf Spot dan Blight memiliki kemiripan visual tinggi (area nekrosis cokelat)
- Analisis ukuran gambar → mayoritas 256×256 px, beberapa outlier hingga 5184×5184 px

### 2. Feature Engineering
- Resize semua gambar ke **224×224 px** (kompatibel dengan pretrained model)
- Data augmentasi yang dipilih berdasarkan relevansi kondisi nyata di lapangan:
  - ✅ Horizontal Flip
  - ✅ Vertical Flip
  - ✅ Rotation 90° (fill_mode=nearest)
  - ✅ Brightness Range [0.5, 1.5]
  - ❌ Zoom (tidak memberikan variasi berarti)
  - ❌ Width/Height Shift (memasukkan background tidak relevan)
- Split: Train / Validation

### 3. Model Baseline — CNN
Arsitektur custom CNN dengan 3 blok Conv2D + GlobalAveragePooling:

| Layer | Detail |
|---|---|
| Conv2D (×3) | Filter bertahap, ReLU activation |
| MaxPooling2D | Stride default |
| GlobalAveragePooling2D | Menggantikan Flatten untuk mengurangi overfitting |
| Dense | 128 unit, ReLU |
| Output | 4 kelas, Softmax |

**Hasil Baseline:**
| Kelas | Recall |
|---|---|
| Blight | 67% |
| Common Rust | 99% |
| Gray Leaf Spot | 77% |
| Healthy | 100% |
| **Overall Accuracy** | **88%** |

### 4. Model Improvement — EfficientNetB0 Transfer Learning
Peningkatan menggunakan **EfficientNetB0** (pretrained ImageNet) dengan tambahan:
- BatchNormalization untuk stabilitas training
- Class weighting untuk mengatasi ketidakseimbangan kelas
- Callbacks: **EarlyStopping** + **ReduceLROnPlateau**
- Early stopping di epoch 30, model terbaik diambil dari epoch 23

---

## 📊 Hasil Model

| Kelas | Baseline CNN | EfficientNetB0 |
|---|---|---|
| Blight | 67% | **90%** ⬆️ |
| Common Rust | 99% | **100%** ⬆️ |
| Gray Leaf Spot | 77% | **82%** ⬆️ |
| Healthy | 100% | **100%** ✅ |
| **Overall Accuracy** | **88%** | **95%** ⬆️ |

**Model terpilih: EfficientNetB0** — peningkatan signifikan terutama pada Blight (+23%) dan Gray Leaf Spot (+5%).

> Tantangan utama: kemiripan visual antara **Gray Leaf Spot** dan **Blight** (area nekrosis cokelat). Beberapa gambar bahkan menunjukkan infeksi ganda yang membuat klasifikasi lebih sulit.

---

## 🚀 Deployment

Aplikasi dideploy sebagai **Streamlit web app** menggunakan Docker di **Hugging Face Spaces**.

**Fitur aplikasi:**
- Upload foto daun jagung
- Hasil klasifikasi penyakit beserta confidence per kelas
- Rekomendasi penanganan per jenis penyakit

🔗 **Live App:** [Lihat di Hugging Face Spaces](https://huggingface.co/) *(update link sesuai repo kamu)*

---

## 🛠️ Tools & Libraries

| Kategori | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras, EfficientNetB0 |
| Data Processing | NumPy, Pandas, PIL |
| Augmentasi | ImageDataGenerator |
| Evaluasi | Scikit-Learn (classification report, confusion matrix) |
| Visualisasi | Matplotlib |
| Deployment | Streamlit, Docker, Hugging Face Spaces |

---

## 👤 Author

**Rezha Aulia**
Hacktiv8 Data Science Bootcamp — Batch FTDS-037-HCK
