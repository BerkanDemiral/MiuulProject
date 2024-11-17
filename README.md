# Akıllı Spor Takibi ve Kalori Tahmin Projesi

## Proje Amacı
Bu proje, spor salonlarında veya bireysel spor aktivitelerinde kullanıcıların egzersiz verilerini analiz ederek yaktıkları kalori miktarını tahmin etmeyi amaçlamaktadır. Çalışma, makine öğrenmesi algoritmaları ve veri ön işleme teknikleri ile desteklenmiştir.

---

## Veri Seti
- **Kaynak:** Kaggle - "Gym Member Exercise Dataset"
- **Açıklama:** Veri seti, spor salonlarındaki üyelerin egzersiz türleri, nabız verileri, seans süreleri gibi çeşitli bilgilerle yaktıkları kalorileri içermektedir.
- **Toplam Satır Sayısı:** 973
- **Toplam Değişken Sayısı:** 15

**Değişkenler:**
- **Age:** Yaş
- **Gender:** Cinsiyet (Erkek/Kadın)
- **Weight (kg):** Kilo
- **Height (m):** Boy
- **Max_BPM:** Maksimum nabız
- **Avg_BPM:** Ortalama nabız
- **Resting_BPM:** Dinlenme nabzı
- **Session_Duration (hours):** Seans süreleri (saat)
- **Calories_Burned:** Yakılan kalori
- **Workout_Type:** Egzersiz türü (Kardiyo, Kuvvet, Yoga, HIIT)
- **Fat_Percentage:** Yağ oranı
- **Water_Intake (liters):** Su tüketimi (litre)
- **Workout_Frequency (days/week):** Egzersiz sıklığı
- **Experience_Level:** Deneyim seviyesi
- **BMI:** Vücut kitle indeksi

---

## Yöntem ve Adımlar

### 1. Veri Hazırlığı
- **Eksik ve Aykırı Değer Analizi:** Boş değer bulunmamış, aykırı değerler "median" ile doldurulmuştur.
- **Değişkenlerin Sınıflandırılması:**
  - Kategorik değişkenler "Label Encoding" ile sayısallaştırılmıştır.
  - Sayısal ve kategorik değişkenler arasındaki korelasyon analiz edilmiştir.
- **Standartleştirme:** Veriler "StandardScaler" ile normalize edilmiştir.

### 2. Makine Öğrenmesi Modelleri
Aşağıdaki algoritmalar kullanılmış ve modellerin performansları karşılaştırılmıştır:

- **Linear Regression**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting Regressor**
- **Random Forest Regressor**
- **Support Vector Regressor (SVR)**

**Performans Değerlendirme:**
- RMSE (Root Mean Square Error)
- R² (Determination Coefficient)
- MAE (Mean Absolute Error)

**Sonuçlar:**
- En iyi performans **Gradient Boosting** algoritması ile elde edilmiştir (RMSE = 0.262, R² = 0.92).

### 3. Parametre Optimizasyonu
Grid Search kullanılarak seçili modellerin hiperparametreleri optimize edilmiştir.

---

## Kullanım Senaryoları
- **Kalori Takibi:** Kullanıcıların yaktıkları kalorileri bireysel olarak takip etmelerini sağlar.
- **Egzersiz Planlaması:** Kalori ihtiyaçlarına göre egzersiz planları oluşturulabilir.
- **Diyet Takibi:** Diyet programları ile entegrasyon sağlanabilir.

---

## Teknik Gereksinimler
- **Python Kütüphaneleri:**
  - NumPy, Pandas, Matplotlib, Seaborn
  - scikit-learn
- **Veri Seti:** `dataset/gym_members_exercise_tracking.csv`

---

## Nasıl Çalıştırılır?

1. Proje dosyalarını indirin.
2. Gerekli kütüphaneleri kurun:
   ```bash
   pip install -r requirements.txt
   ```
3. Proje kodlarını çalıştırın:
   ```bash
   python projectCode.py
   ```
4. Sonuçlar ve grafikler çıktı olarak görüntülenecektir.

---

## Geliştirici
Berkan Demiral

