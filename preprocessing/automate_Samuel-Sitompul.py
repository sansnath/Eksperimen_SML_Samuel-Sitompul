"""
automate_Samuel-Sitompul.py
---------------------------
Script otomasi preprocessing dataset Loan Approval.
Mengonversi tahapan eksperimen pada notebook menjadi fungsi yang dapat
dijalankan secara otomatis untuk menghasilkan data yang siap dilatih.

Penggunaan:
    python automate_Samuel-Sitompul.py \
        --input  ../loan_approval_dataset_raw.csv \
        --output loan_approval_dataset_preprocessing.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── 1. Data Loading ──────────────────────────────────────────────────────────

def load_data(input_path: str) -> pd.DataFrame:
    """Memuat dataset dari path CSV yang diberikan."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
    df = pd.read_csv(input_path)
    # Hapus spasi di nama kolom
    df.columns = df.columns.str.strip()
    # Hapus spasi di nilai string
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ── 2. Cleaning ───────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data: hapus duplikat, drop kolom tidak relevan."""
    # Hapus duplikat
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Duplikat dihapus: {before - len(df)} baris")

    # Drop kolom loan_id (hanya identifier, tidak informatif untuk model)
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])
        print("[INFO] Kolom 'loan_id' dihapus")

    # Periksa missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.dropna()
        print(f"[INFO] Baris dengan missing values dihapus: {missing}")
    else:
        print("[INFO] Tidak ada missing values")

    return df


# ── 3. Encoding Kategorikal ──────────────────────────────────────────────────

def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Melakukan Label Encoding pada kolom kategorikal."""
    label_encoders = {}
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"[INFO] Kolom kategorikal yang di-encode: {cat_cols}")

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  - {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df, label_encoders


# ── 4. Feature Scaling ───────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame, target_col: str = "loan_status") -> tuple[pd.DataFrame, StandardScaler]:
    """Melakukan StandardScaler pada kolom numerik (kecuali target)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    print(f"[INFO] Kolom numerik yang di-scale: {num_cols}")

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler


# ── 5. Train-Test Split ──────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Membagi dataset menjadi data latih dan data uji."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── 6. Simpan Hasil ───────────────────────────────────────────────────────────

def save_preprocessed(df: pd.DataFrame, output_path: str) -> None:
    """Menyimpan dataset yang sudah diproses ke CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset preprocessing disimpan: {output_path}")


# ── Fungsi Utama ──────────────────────────────────────────────────────────────

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Fungsi utama untuk melakukan preprocessing secara otomatis.

    Parameters
    ----------
    input_path  : str  Path ke dataset raw (.csv)
    output_path : str  Path untuk menyimpan dataset yang sudah diproses

    Returns
    -------
    pd.DataFrame  Dataset yang sudah siap dilatih
    """
    print("=" * 55)
    print("  AUTOMATE PREPROCESSING — Samuel Sitompul")
    print("=" * 55)

    # Langkah 1: Load
    df = load_data(input_path)

    # Langkah 2: Clean
    df = clean_data(df)

    # Langkah 3: Encode kategorikal
    df, _ = encode_categorical(df)

    # Langkah 4: Scale fitur numerik
    df, _ = scale_features(df, target_col="loan_status")

    # Langkah 5: Simpan hasil preprocessing
    save_preprocessed(df, output_path)

    print("=" * 55)
    print("  PREPROCESSING SELESAI!")
    print(f"  Output shape : {df.shape}")
    print("=" * 55)

    return df


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate Preprocessing — Loan Approval Dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../loan_approval_dataset_raw.csv",
        help="Path ke dataset raw",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="loan_approval_dataset_preprocessing.csv",
        help="Path output dataset preprocessing",
    )
    args = parser.parse_args()

    preprocess_data(args.input, args.output)
