# Daegu Apartment Price Prediction Machine Learning

Proyek ini memuat end-to-end Machine Learning project untuk memprediksi harga apartemen yaitu `SalePrice` di Daegu, Korea Selatan - berdasarkan beberapa 10 feature berpengaruh terhadap harga apartemen. Pada proyek ini, akan dijabarkan secara bertahap dimulai dari memahami konteks bisnis → melakukan inspeksi terhadap data → Exploratory Data Analysis → Data Preprocessing → Feature Engineering & Selection → Machine Learning Modeling → Model Evaluation → Kesimpulan dan Rekomendasi.

## Business Problem

Pemilik apartemen dan agen platform di Daegu saat ini menentukan harga jual apartemen berdasarkan perkiraan/feeling, tanpa acuan berbasis data. Harga terlalu tinggi membuat unit sulit terjual; harga terlalu rendah membuat pemilik kehilangan potensi keuntungan. Proyek ini membangun model regresi yang memberikan **harga usulan (suggested price)** saat apartemen akan dijual, mengubah proses penentuan harga dari perkiraan menjadi keputusan berbasis data.

**Goal:** memprediksi `SalePrice` dengan **MAPE ≤ 20%** pada data test — ✅ tercapai (19,82%).

## Dataset

`data_daegu_apartment.csv` memiliki 2.701 listing apartemen unik (setelah menghapus 1.422 baris duplikat persis) dengan feature berikut:

| Kolom | Deskripsi |
|---|---|
| `HallwayType` | Tipe desain akses apartemen (`corridor`, `terraced`, `mixed`) |
| `TimeToSubway` | Kategori waktu tempuh ke stasiun subway terdekat |
| `SubwayStation` | Nama stasiun subway terdekat (proxy lokasi) |
| `N_FacilitiesNearBy(ETC)` | Jumlah fasilitas umum lain di sekitar |
| `N_FacilitiesNearBy(PublicOffice)` | Jumlah kantor publik di sekitar |
| `N_SchoolNearBy(University)` | Jumlah universitas di sekitar |
| `N_Parkinglot(Basement)` | Kapasitas parkir basement |
| `YearBuilt` | Tahun bangunan didirikan |
| `N_FacilitiesInApt` | Jumlah fasilitas di dalam gedung apartemen |
| `Size(sqf)` | Luas unit (sqft) |
| `SalePrice` | **Target** — harga jual dalam Won |

## Struktur Proyek

```
.
├── daegu_apartment_regression.ipynb   # pipeline lengkap: EDA -> cleaning -> modeling -> evaluation -> export
├── data_daegu_apartment.csv           # dataset mentah
├── gb_model_baseline.pkl              # model final terlatih (pickled sklearn Pipeline)
└── README.md
```

## Metodologi

1. **EDA** — distribusi data numerik dan kategori, heatmap korelasi dan relasi `SalePrice` dengan feature yang ada
2. **Data preprocessing** — menghapus 1.422 baris duplikat persis; menginvestigasi outlier (berdasarkan IQR) pada `Size(sqf)` dan `SalePrice`, dan dengan sengaja **mempertahankannya** setelah dikonfirmasi bahwa outlier tersebut merepresentasikan segmen bangunan mewah/terraced yang nyata, bukan kesalahan data
3. **Feature engineering** — `Age` (relatif terhadap tahun bangunan terbaru) dan `NearbyFacilitiesScore` (kombinasi dari 3 feature "jumlah fasilitas sekitar" yang saling berkorelasi tinggi/multikolinearitas tinggi)
4. **Feature selection** — analisis VIF untuk memeriksa multikolinearitas dan menjustifikasi susunan feature hasil engineering
5. **Modeling** — membandingkan Linear Regression, Ridge, Random Forest, Gradient Boosting, dan XGBoost, masing-masing dibungkus dalam `ColumnTransformer` + `Pipeline` (StandardScaler untuk numerik, OneHotEncoder untuk kategorikal), dipilih melalui 5-fold cross-validation
6. **Evaluation** — MAE, RMSE, MAPE, R², ditambah metrik bisnis custom (persentase prediksi yang berada dalam ±10% dari harga aktual)
7. **Kesimpulan dan Rekomendasi** - Memberikan perbandingan model pada data training dan data test serta memberikan highlights dan rekomendasi yang dapat dilakukan kedepannya untuk memperbaiki model lebih baik
8. **Model export** — pipeline final disimpan (pickle) untuk digunakan kembali

## Hasil

| Metric | CV (Training, mean) | Test Evaluation | Selisih |
|---|---|---|---|
| RMSE | 45,888.35 Won | 47,752 Won | +1,863.65 Won |
| MAE | 37,121.21 Won | 38,876 Won | +1,754.79 Won |
| MAPE | 19.18% | 19.82% | +0.64% |
| R² | 0.8100 | 0.7853 | -0.0247 |

 Model yang disimpan (`gb_model_baseline.pkl`) menggunakan baseline tanpa tuning

## Rekomendasi Penggunaan

Rekomendasi penggunaan hasil Machine Learning: tampilkan hasil prediksi sebagai **harga usulan**, bukan angka tetap, dan arahkan pemilik/agen untuk ditinjau secara manual untuk memberi tambahan keyakinan terhadap harga yang akan dipasang.

## Cara Menjalankan

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels scipy
jupyter notebook daegu_apartment_regression.ipynb
```

## Memuat Model yang Sudah Disimpan

```python
import pickle

with open("gb_model_baseline.pkl", "rb") as f:
    model = pickle.load(f)

# Masukkan kolom feature mentah (belum diproses) -- pipeline akan
# menangani scaling/encoding secara otomatis di dalamnya.
predicted_price = model.predict(new_listing_df)
```

## Langkah Selanjutnya

- Melakukan hyperparameter tuning untuk mengecilkan gap MAPE
- Melacak hasil transaksi aktual (harga terjual, lama waktu listing/days-on-market) untuk memvalidasi model terhadap perilaku penjualan yang sesungguhnya, bukan hanya harga listing
- Menambahkan feature-feature lain yang dapat menangkap lebih banyak faktor dalam penentuan harga apartemen
- Mengulang investigasi baris duplikat untuk memastikan keputusan pembersihan memang tepat

## Penulis

Kemal Aditya Permana - JCAIEAH004 - Purwadhika Capstones Project Module 2
