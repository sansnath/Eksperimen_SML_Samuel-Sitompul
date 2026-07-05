# Eksperimen_SML_Samuel-Sitompul

Repositori eksperimen Machine Learning untuk dataset **Loan Approval Prediction**.

## Struktur Folder

```
Eksperimen_SML_Samuel-Sitompul/
├── loan_approval_dataset_raw.csv          # Dataset mentah
└── preprocessing/
    ├── Eksperimen_Samuel-Sitompul.ipynb   # Notebook eksperimen lengkap
    ├── automate_Samuel-Sitompul.py        # Script otomasi preprocessing
    └── loan_approval_dataset_preprocessing.csv  # Dataset siap latih
```

## Dataset

- **Sumber**: [Kaggle — Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Ukuran**: 4.269 baris × 13 kolom
- **Target**: `loan_status` (Approved / Rejected)

## Cara Menjalankan

### 1. Jalankan Notebook
Buka `preprocessing/Eksperimen_Samuel-Sitompul.ipynb` di Jupyter Notebook atau Google Colab.

### 2. Jalankan Automate Script
```bash
python preprocessing/automate_Samuel-Sitompul.py \
    --input  loan_approval_dataset_raw.csv \
    --output preprocessing/loan_approval_dataset_preprocessing.csv
```

## Tahapan Preprocessing
1. Load dataset & strip whitespace
2. Drop kolom `loan_id`
3. Label Encoding (education, self_employed, loan_status)
4. StandardScaler pada fitur numerik
5. Train-Test Split (80/20)
