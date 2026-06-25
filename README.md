# 📊 NEET-Sighted Indonesia 2024 — Streamlit Dashboard

Dashboard analisis data NEET (Not in Education, Employment, or Training) untuk 38 provinsi di Indonesia.

---

## 🗂️ Struktur Proyek

```
neet_dashboard/
├── app.py                     ← File utama aplikasi Streamlit
├── requirements.txt           ← Daftar dependensi Python
├── README.md                  ← Panduan ini
└── data/
    ├── master_data_neet_clean.csv       ← Data utama 38 provinsi
    ├── hasil_clustering_neet.csv        ← Hasil K-Means clustering
    └── indonesia_38_provinsi.geojson    ← Peta GeoJSON Indonesia
```

---

## ⚙️ Cara Menjalankan

### 1. Prasyarat
Pastikan Python 3.9+ sudah terinstall di komputer kamu.

### 2. Clone / Download Proyek
Letakkan semua file sesuai struktur di atas.

### 3. Install Dependensi

Buka terminal, masuk ke folder `neet_dashboard`, lalu jalankan:

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install streamlit pandas numpy plotly
```

### 4. Jalankan Dashboard

```bash
streamlit run app.py
```

Dashboard akan otomatis terbuka di browser di alamat:
```
http://localhost:8501
```

---

## 📋 Fitur Dashboard

### 📊 Halaman 1 — Overview & Distribusi
- **4 KPI Cards**: Rata-rata nasional, tertinggi, terendah, provinsi kritis
- **Choropleth Map**: Peta Indonesia dengan gradasi warna berdasarkan tingkat NEET
- **Bar Chart Interaktif**: Filter per wilayah (Jawa, Sumatera, Kalimantan, dll.)
- **Box Plot & Distribusi**: Sebaran NEET per pulau
- **Scatter Plot**: NEET vs IPM dengan warna cluster

### 📉 Halaman 2 — Regresi & Korelasi
- **Tabel Regresi OLS**: Koefisien, std. error, t-stat, signifikansi
- **Diverging Bar Chart**: Korelasi tiap variabel dengan NEET
- **Heatmap Matriks Korelasi**: Visualisasi 5×5 correlation matrix
- **Coefficient Plot**: Estimasi β dengan confidence interval 95%
- **Scatter Plot Interaktif**: Pilih TPT / IPM / SMK_Komp untuk diplot vs NEET
- **Actual vs Predicted**: Validasi model dengan titik warna per cluster

### 🔵 Halaman 3 — Clustering K-Means
- **3 Cluster Cards**: Ringkasan statistik tiap cluster
- **Peta Cluster**: GeoJSON choropleth dengan warna hijau/kuning/merah
- **PCA Scatter Plot**: Bukti separasi 3 cluster dalam ruang PC1-PC2
- **Radar Chart**: Profil 5 dimensi tiap cluster
- **Tabel Provinsi**: Filter + search, sortable per cluster

### 🎯 Halaman 4 — Simulasi & Rekomendasi
- **4 Temuan Utama**: Card ringkasan insight analisis
- **Policy Simulator Interaktif**: 3 slider (TPT, IPM, SMK) → proyeksi NEET
- **Grafik Simulasi**: Line chart log(NEET) vs % SMK berkomputerisasi
- **Rekomendasi per Provinsi**: Pilih provinsi → tampil rekomendasi spesifik + radar chart profil

---

## 📦 Dependensi

| Library | Versi Minimum | Kegunaan |
|---------|--------------|----------|
| streamlit | 1.28.0 | Framework dashboard web |
| pandas | 2.0.0 | Manipulasi data CSV |
| numpy | 1.24.0 | Kalkulasi numerik |
| plotly | 5.17.0 | Semua visualisasi interaktif |

---

## 🗃️ Tentang Data

- **Sumber**: BPS, Kemendikbud, Kemnaker 2024
- **Cakupan**: 38 provinsi Indonesia (termasuk 4 DOB Papua)
- **Variabel Utama**:
  - `Target_NEET`: Persentase pemuda NEET (15–24 tahun)
  - `TPT`: Tingkat Pengangguran Terbuka
  - `IPM`: Indeks Pembangunan Manusia
  - `SMK_Komp`: % SMK yang memiliki komputer
  - `Miskin`: Persentase penduduk miskin
  - `cluster`: Hasil K-Means (1=Hijau, 2=Kuning, 3=Merah)

---

## 🔧 Tips Troubleshooting

**Port sudah dipakai?**
```bash
streamlit run app.py --server.port 8502
```

**Data tidak terbaca?**
Pastikan kamu menjalankan `streamlit run app.py` dari dalam folder `neet_dashboard/`, bukan dari folder lain.

**Peta tidak muncul?**
Pastikan file `indonesia_38_provinsi.geojson` ada di folder `data/` dan koneksi internet tersedia (untuk Plotly render GeoJSON).
