# AdaBoost ile Diyabet Tahmini

Pima Indians Diabetes veri seti üzerinde, hastanın temel klinik ölçümlerinden diyabet olup olmadığını tahmin eden bir ikili sınıflandırma çalışması. Veri setindeki fizyolojik olarak imkânsız sıfır değerleri eksik veri olarak ele alınıp medyan ve KNN tabanlı doldurma ile giderildikten sonra, **AdaBoostClassifier** ve karşılaştırma tabanı olarak **RidgeClassifier** eğitilmiştir. Tüm akış tek bir Jupyter Notebook içinde, uçtan uca ve tekrar üretilebilir biçimde kurgulanmıştır.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-AdaBoost-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Veri Seti

Kullanılan veri seti [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (UCI / Kaggle). Depoda `16-diabetes.csv` olarak yer alır.

- **Gözlem sayısı:** 768 satır
- **Sütun sayısı:** 9 (8 özellik + 1 hedef)
- **Hedef değişken:** `Outcome` — 1: diyabet, 0: diyabet değil
- **Sınıf dağılımı:** %65,10 → `0` · %34,90 → `1` (dengesiz, ancak ağır olmayan bir dağılım)
- **Eksik değer:** `df.info()` çıktısında hiçbir sütunda null yok; ancak bazı sütunlardaki `0` değerleri gerçekte eksik ölçümü temsil ediyor (aşağıya bakınız).

### Özellikler

| Sütun | Açıklama | Tip |
|---|---|---|
| `Pregnancies` | Gebelik sayısı | int64 |
| `Glucose` | Plazma glikoz konsantrasyonu | int64 |
| `BloodPressure` | Diyastolik kan basıncı (mm Hg) | int64 |
| `SkinThickness` | Triseps deri kıvrım kalınlığı (mm) | int64 |
| `Insulin` | 2 saatlik serum insülin (mu U/ml) | int64 |
| `BMI` | Vücut kitle indeksi | float64 |
| `DiabetesPedigreeFunction` | Diyabet soy geçmişi fonksiyonu | float64 |
| `Age` | Yaş (yıl) | int64 |
| `Outcome` | **Hedef** — diyabet durumu (0/1) | int64 |

### Gizli eksik veri: sıfır değerleri

Fizyolojik olarak sıfır olamayacak sütunlarda tespit edilen sıfır sayıları (notebook çıktısı):

| Sütun | Sıfır sayısı | Oran (768 üzerinden) |
|---|---:|---:|
| `Insulin` | 374 | %48,7 |
| `SkinThickness` | 227 | %29,6 |
| `BloodPressure` | 35 | %4,6 |
| `BMI` | 11 | %1,4 |
| `Glucose` | 5 | %0,7 |

### Hedef ile korelasyon

`df.corr()['Outcome']` sonuçları (sıfırlar `NaN` olarak işaretlendikten sonra):

| Özellik | Korelasyon |
|---|---:|
| `Glucose` | 0,495 |
| `BMI` | 0,314 |
| `Insulin` | 0,303 |
| `SkinThickness` | 0,259 |
| `Age` | 0,238 |
| `Pregnancies` | 0,222 |
| `DiabetesPedigreeFunction` | 0,174 |
| `BloodPressure` | 0,171 |

Glikoz, hedefle açık ara en güçlü ilişkiye sahip değişken.

---

## Yöntem / İş Akışı

### 1. Keşifsel veri analizi (EDA)
- `df.head()`, `df.info()`, `df.describe()` ile yapısal inceleme
- `Outcome` sınıf dağılımı için `countplot` ve yüzdelik oranlar
- Tüm sayısal değişkenler için histogramlar (`bins=20`)
- Korelasyon matrisi ısı haritası (`seaborn.heatmap`, `annot=True`)
- Her özellik için `Outcome` kırılımında boxplot'lar

### 2. Eksik veri işleme
1. `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` sütunlarındaki `0` değerleri `np.nan` ile değiştirilir.
2. Veri **önce** eğitim/test olarak ayrılır — doldurma istatistikleri yalnızca eğitim kümesinden öğrenilir, böylece veri sızıntısı (data leakage) önlenir.
3. `Glucose`, `BloodPressure`, `BMI`, `SkinThickness` için eğitim kümesinden medyan değerleri hesaplanır ve aynı medyanlar test kümesine uygulanır.
4. Kalan eksik değerler `KNNImputer(n_neighbors=5)` ile doldurulur: imputer **yalnızca eğitim kümesinde** `fit` edilir, test kümesine `transform` uygulanır.

> Not: Ölçeklendirme (`StandardScaler`) kodda yer alıyor ancak yorum satırına alınmış durumda — AdaBoost ağaç tabanlı olduğu için ölçek duyarsızdır.

### 3. Eğitim/test ayrımı

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15
)
```

→ **614** eğitim, **154** test örneği.

### 4. Modeller ve hiperparametreler

| Model | Hiperparametreler |
|---|---|
| `AdaBoostClassifier` | scikit-learn varsayılanları |
| `RidgeClassifier` | `alpha=10`, `solver="auto"`, `random_state=42` |

Modeller bir sözlük (`dict`) üzerinden döngüyle eğitilir; metrikler `calculate_model_metrics(true, predicted)` adlı modüler bir yardımcı fonksiyonla (`classification_report`, `accuracy_score`, `confusion_matrix`) hem eğitim hem test kümesi için raporlanır.

---

## Sonuçlar

Aşağıdaki tüm değerler notebook'un çalıştırılmış çıktısından alınmıştır.

### AdaBoostClassifier

| Küme | Accuracy | Precision (sınıf 1) | Recall (sınıf 1) | F1 (sınıf 1) |
|---|---:|---:|---:|---:|
| Eğitim | **0,8046** | 0,77 | 0,66 | 0,71 |
| Test | **0,7662** | 0,60 | 0,63 | 0,62 |

**Test karmaşıklık matrisi (confusion matrix):**

|  | Tahmin: 0 | Tahmin: 1 |
|---|---:|---:|
| **Gerçek: 0** | 89 | 19 |
| **Gerçek: 1** | 17 | 29 |

Ağırlıklı ortalama precision / recall / F1 (test): 0,77 / 0,77 / 0,77

### RidgeClassifier (karşılaştırma tabanı)

| Küme | Accuracy | Precision (sınıf 1) | Recall (sınıf 1) | F1 (sınıf 1) |
|---|---:|---:|---:|---:|
| Eğitim | **0,7720** | 0,74 | 0,57 | 0,64 |
| Test | **0,7662** | 0,61 | 0,59 | 0,60 |

**Test karmaşıklık matrisi:**

|  | Tahmin: 0 | Tahmin: 1 |
|---|---:|---:|
| **Gerçek: 0** | 91 | 17 |
| **Gerçek: 1** | 19 | 27 |

### Değerlendirme

- İki model de test kümesinde **%76,6 doğruluk** ile aynı seviyede sonuç veriyor.
- AdaBoost, pozitif sınıfta (diyabet) daha yüksek recall (0,63 vs 0,59) ve daha yüksek F1 (0,62 vs 0,60) elde ediyor. Tarama amaçlı tıbbi bir problemde yanlış negatifleri azaltmak kritik olduğundan bu fark AdaBoost lehine anlamlı.
- AdaBoost'un eğitim doğruluğu (0,8046) ile test doğruluğu (0,7662) arasındaki ~4 puanlık fark, sınırlı ve kabul edilebilir düzeyde bir aşırı öğrenme (overfitting) işareti.
- Her iki modelde de asıl darboğaz azınlık sınıfının recall değeri; sınıf ağırlıklandırma, eşik optimizasyonu veya hiperparametre araması (`GridSearchCV` notebook'a import edilmiş, sonraki adım için hazır) doğal iyileştirme yönleri.

---

## Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.x
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/<kullanici-adi>/diabetes-prediction.git
cd diabetes-prediction

# 2. Sanal ortam oluşturun (önerilir)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Bağımlılıkları kurun
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Notebook'u açın
jupyter notebook 16-AdaboostClasifier.ipynb
```

Notebook, `16-diabetes.csv` dosyasını aynı dizinden okur. Hücreleri baştan sona sırayla çalıştırmanız yeterlidir.

---

## Dosya Yapısı

```
diabetes-prediction/
├── 16-AdaboostClasifier.ipynb   # EDA, ön işleme, model eğitimi ve değerlendirme
├── 16-diabetes.csv              # Pima Indians Diabetes veri seti (768 x 9)
├── .gitignore                   # Jupyter / Python / editör çıktıları
├── LICENSE                      # MIT lisansı
└── README.md
```

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

Veri seti: UCI Machine Learning Repository / Kaggle — [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
