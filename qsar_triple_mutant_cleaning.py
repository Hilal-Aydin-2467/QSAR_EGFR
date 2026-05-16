"""
QSAR Veri Temizleme — EGFR L858R/T790M/C797S Triple Mutant
===========================================================
Atıf: Li et al. Pharmaceuticals 2025, 18, 1092
DOI : 10.3390/ph18081092

Atıftaki çalışmanın veri derleme mantığına uygun olarak:
  • Sadece EGFR L858R/T790M/C797S triple mutant binding assay verileri alındı
  • Homo sapiens hedef organizma
  • Standard Relation = '=' (kesin deneysel değerler, sansürlüler çıkarıldı)
  • Assay Type = 'B' (binding assay)
  • Data Validity Comment olmayan satırlar (hatalı/uç değerler çıkarıldı)
  • pChEMBL Value ve SMILES boş olanlar çıkarıldı
  • Aynı moleküle ait birden fazla ölçüm varsa pIC50 ORTALAMASI alındı

Çıktı sütunları:
  Molecule ChEMBL ID | Smiles | pIC50 | n_measurements
"""

import pandas as pd
import numpy as np

# ─── GİRDİ / ÇIKTI ───────────────────────────────────────────────────────────
INPUT_FILE = r"C:/Users/LENOVO/Documents/Qsar_EGFR/DOWNLOAD-e_fT7fM1sBJ4f_23FUhjlMsjn-3P-DpoxrkXG5AFUiI_eq_.csv"
OUTPUT_FILE = r"C:/Users/LENOVO/Documents/Qsar_EGFR/egfr_qsar_cleaned.csv"
# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 62

print(SEP)
print("  QSAR VERİ TEMİZLEME — EGFR L858R/T790M/C797S")
print(SEP)

# ── 1. OKU ───────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, sep=";")
print(f"\n[1] Ham veri okundu          : {len(df):>6,} satır")

# ── 2. TARGET ORGANISM ────────────────────────────────────────────────────────
df = df[df["Target Organism"] == "Homo sapiens"].copy()
print(f"[2] Target Organism filtresi  : {len(df):>6,} satır")

# ── 3. STANDARD TYPE = IC50 ───────────────────────────────────────────────────
df = df[df["Standard Type"] == "IC50"].copy()
print(f"[3] Standard Type = IC50      : {len(df):>6,} satır")

# ── 4. TRIPLE MUTANT FİLTRESİ ────────────────────────────────────────────────
#   ChEMBL'de triple mutant 'L858R,T790M,C797S' olarak geçiyor
#   Atıftaki makale bu mutasyon kombinasyonunu hedefliyor
TARGET_MUTATION = "L858R,T790M,C797S"
before = len(df)
df = df[df["Assay Variant Mutation"] == TARGET_MUTATION].copy()
print(f"[4] Triple mutant filtresi    : {len(df):>6,} satır  "
      f"({before - len(df):,} satır çıkarıldı)")

# ── 5. STANDARD RELATION = '=' ────────────────────────────────────────────────
#   '>' ve '<' sansürlü veridir (kesin IC50 bilinmez → SVR için uygun değil)
before = len(df)
print(f"\n     Standard Relation dağılımı (mutant filtre sonrası):")
for rel, cnt in df["Standard Relation"].value_counts().items():
    flag = "✓ TUTULACAK" if rel == "'='" else "✗ ÇIKARILACAK"
    print(f"       {rel:6s}  {cnt:4d} satır  {flag}")
df = df[df["Standard Relation"] == "'='"].copy()
print(f"[5] Standard Relation = '='   : {len(df):>6,} satır  "
      f"({before - len(df):,} sansürlü satır çıkarıldı)")

# ── 6. ASSAY TYPE = 'B' ───────────────────────────────────────────────────────
before = len(df)
df = df[df["Assay Type"] == "B"].copy()
print(f"[6] Assay Type = 'B'          : {len(df):>6,} satır  "
      f"({before - len(df):,} satır çıkarıldı)")

# ── 7. DATA VALIDITY COMMENT ─────────────────────────────────────────────────
before = len(df)
invalid = df["Data Validity Comment"].notna()
if invalid.sum() > 0:
    print(f"\n     Data Validity Comment (geçersiz veri):")
    for cat, cnt in df.loc[invalid, "Data Validity Comment"].value_counts().items():
        print(f"       '{cat}': {cnt} satır")
df = df[df["Data Validity Comment"].isna()].copy()
print(f"[7] Validity Comment filtresi : {len(df):>6,} satır  "
      f"({before - len(df):,} satır çıkarıldı)")

# ── 8. pChEMBL VALUE NaN ─────────────────────────────────────────────────────
before = len(df)
df = df[df["pChEMBL Value"].notna()].copy()
df["pChEMBL Value"] = pd.to_numeric(df["pChEMBL Value"], errors="coerce")
df = df[df["pChEMBL Value"].notna()].copy()
print(f"[8] pChEMBL Value NaN silindi : {len(df):>6,} satır  "
      f"({before - len(df):,} satır çıkarıldı)")

# ── 9. SMILES NaN ────────────────────────────────────────────────────────────
before = len(df)
df = df[df["Smiles"].notna()].copy()
print(f"[9] SMILES NaN silindi        : {len(df):>6,} satır  "
      f"({before - len(df):,} satır çıkarıldı)")

# ── 10. DUPLİKAT ORTALAMA ────────────────────────────────────────────────────
print(f"\n[10] Duplikat analizi:")
n_total  = len(df)
n_uniq   = df["Molecule ChEMBL ID"].nunique()
print(f"     Toplam satır          : {n_total:,}")
print(f"     Benzersiz molekül     : {n_uniq:,}")
print(f"     Duplikat satır        : {n_total - n_uniq:,}")

# Aynı ChEMBL ID → SMILES'ın ilki + pIC50 ortalaması
smiles_map = df.groupby("Molecule ChEMBL ID")["Smiles"].first()

agg = (
    df.groupby("Molecule ChEMBL ID")
    .agg(
        pIC50          = ("pChEMBL Value", "mean"),
        pIC50_std      = ("pChEMBL Value", "std"),   # ölçümler arası tutarsızlık
        n_measurements = ("pChEMBL Value", "count"),
    )
    .reset_index()
)
agg["pIC50_std"] = agg["pIC50_std"].fillna(0.0)
agg["Smiles"] = agg["Molecule ChEMBL ID"].map(smiles_map)

# Sütun sırası
df_clean = agg[[
    "Molecule ChEMBL ID",
    "Smiles",
    "pIC50",
    "pIC50_std",
    "n_measurements"
]].copy()

print(f"\n     Ortalama sonrası      : {len(df_clean):,} benzersiz molekül")
print(f"\n     n_measurements dağılımı:")
vc = df_clean["n_measurements"].value_counts().sort_index()
for n, c in vc.items():
    bar = "█" * min(c, 30)
    print(f"       {n:2d} ölçüm  →  {c:3d} molekül  {bar}")

# ── 11. pIC50 İSTATİSTİKLERİ ─────────────────────────────────────────────────
print(f"\n[11] pIC50 dağılım istatistikleri (log10 ölçeği):")
stats = df_clean["pIC50"].describe()
print(f"     Ortalama   : {stats['mean']:.3f}")
print(f"     Std        : {stats['std']:.3f}")
print(f"     Min        : {stats['min']:.3f}")
print(f"     25%        : {stats['25%']:.3f}")
print(f"     Medyan     : {stats['50%']:.3f}")
print(f"     75%        : {stats['75%']:.3f}")
print(f"     Max        : {stats['max']:.3f}")

# pIC50 aralık grupları
bins   = [0, 5, 6, 7, 8, 9, 12]
labels = ["<5", "5-6", "6-7", "7-8", "8-9", ">9"]
df_clean["pIC50_grup"] = pd.cut(df_clean["pIC50"], bins=bins, labels=labels)
print(f"\n     pIC50 aralık dağılımı:")
for lbl, cnt in df_clean["pIC50_grup"].value_counts().sort_index().items():
    bar = "█" * min(cnt, 40)
    print(f"       pIC50 {lbl:5s}  {cnt:4d} mol  {bar}")
df_clean = df_clean.drop(columns=["pIC50_grup"])

# ── 12. KAYDET ────────────────────────────────────────────────────────────────
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\n[12] Dosya kaydedildi: {OUTPUT_FILE}")

print(f"\n{SEP}")
print("  ÖZET")
print(SEP)
print(f"  Ham veri                : 25,758 satır (tüm EGFR ChEMBL)")
print(f"  Triple mutant ham       :    629 satır (L858R/T790M/C797S)")
print(f"  Filtreler sonrası       :    {len(df_clean):>3} benzersiz molekül")
print(f"  Hedef değişken          : pIC50 = -log10(IC50[M])")
print(f"  Mutasyon                : EGFR L858R/T790M/C797S")
print(f"  Kayıt                   : {OUTPUT_FILE.split('/')[-1]}")
print(SEP)

print("\nİlk 5 satır:")
print(df_clean.head(5).to_string(index=False))

print("""
─────────────────────────────────────────────────────────────
Sonraki adım — SVR pipeline:
  1. SMILES → Morgan Fingerprint (RDKit, radius=2, 2048 bit)
  2. Train/Test split (80:20 veya 4:1, atıfta olduğu gibi)
  3. Feature scaling (StandardScaler)
  4. SVR (kernel=rbf/poly, C ve epsilon optimizasyonu)
  5. Validasyon: R2, RMSE, Q2_LOO, Q2_5fold, CCC
─────────────────────────────────────────────────────────────
""")
