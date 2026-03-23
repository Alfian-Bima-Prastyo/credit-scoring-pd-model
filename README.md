# Credit Risk — Probability of Default Modeling

Portofolio project end-to-end PD model menggunakan dataset Home Credit.

## Tujuan
Memprediksi probabilitas default (PD) seorang peminjam dan mensimulasikan kebijakan kredit berdasarkan cutoff PD.

## Dataset
- Sumber: [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)
- File: `application_train.csv`
- Target: `TARGET` (1 = default, 0 = non-default)
- Class imbalance: ~8% default

## Metodologi

### Preprocessing
- Drop kolom dengan missing > 80%
- Capping income outlier di percentile ke-99
- Flag anomaly `DAYS_EMPLOYED = 365243`
- Impute dan flag missing `EXT_SOURCE_1/2/3`
- Feature engineering: ratio kredit, income, usia

### Model
| Model | Validation AUC |
|---|---|
| Logistic Regression | 0.74632 |
| LightGBM | 0.76259 |
| XGBoost | 0.76318 |

Model terpilih: **LightGBM** — selisih AUC dengan XGBoost hanya 0.00059, tidak signifikan secara bisnis, namun LightGBM lebih cepat dan efisien secara memori.

### Evaluasi
| Metrik | Nilai |
|---|---|
| Test AUC | 0.76732 |
| Gini | 0.53464 |
| KS Statistic | 0.40023 |
| Brier Score OOT | 0.081089 |
| Score PSI | 0.084126 |

### Validasi tambahan
- Stratified K-Fold (5 fold) pada training set
- Out-of-Time (OOT) validation menggunakan `DAYS_ID_PUBLISH`
- Population Stability Index (PSI) fitur top 20
- Calibration curve dan Brier Score
- Fairness check by age group
- Cutoff policy simulation + stress testing (Base, High LGD, Low Interest)

## Cara menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur folder
```
├── app.py
├── predict.py
├── preprocessing.py
├── requirements.txt
├── model/
│   ├── final_model.pkl
│   ├── income_cap.pkl
│   └── feature_columns.pkl
└── notebooks/
    └── credit_scoring.ipynb
```
