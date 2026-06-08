"""
Exploratory Data Analysis of Kaggle Sirio-Libanes ICU Prediction Dataset
========================================================================
Input: /home/marko-b2/COVID_DATASETS/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx
Output: /home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/eda_report/

Goal: Quick audit before ML training. No preprocessing yet.
Author: Marko Živanović
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
INPUT = Path('/home/marko-b2/COVID_DATASETS/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/eda_report')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*70)
print('SIRIO-LIBANES ICU PREDICTION — EXPLORATORY DATA ANALYSIS')
print('='*70)

# Load
df = pd.read_excel(INPUT)
print(f'\nShape: {df.shape[0]} rows x {df.shape[1]} columns')

# 1. Patient-level view
n_patients = df['PATIENT_VISIT_IDENTIFIER'].nunique()
print(f'Unique patients: {n_patients}')
print(f'Records per patient (avg): {len(df)/n_patients:.2f}')

# 2. WINDOW variable
print(f'\nWINDOW values: {df["WINDOW"].value_counts().to_dict()}')

# 3. Outcome distribution — overall and per patient
print('\n--- OUTCOME: ICU ---')
print(f'Total rows ICU=1: {(df["ICU"]==1).sum()} ({(df["ICU"]==1).mean()*100:.1f}%)')
print(f'Total rows ICU=0: {(df["ICU"]==0).sum()} ({(df["ICU"]==0).mean()*100:.1f}%)')

# Per-patient ICU (ever went to ICU)
patient_icu = df.groupby('PATIENT_VISIT_IDENTIFIER')['ICU'].max()
print(f'Patients who EVER went to ICU: {(patient_icu==1).sum()} ({(patient_icu==1).mean()*100:.1f}%)')
print(f'Patients who NEVER went to ICU: {(patient_icu==0).sum()} ({(patient_icu==0).mean()*100:.1f}%)')

# 4. Demographics
print('\n--- DEMOGRAPHICS ---')
print(f'AGE_ABOVE65 distribution:\n{df["AGE_ABOVE65"].value_counts()}')
print(f'\nGENDER distribution:\n{df["GENDER"].value_counts()}')

# 5. Comorbidities
print('\n--- COMORBIDITIES ---')
como = ['DISEASE GROUPING 1', 'DISEASE GROUPING 2', 'DISEASE GROUPING 3',
        'DISEASE GROUPING 4', 'DISEASE GROUPING 5', 'DISEASE GROUPING 6',
        'HTN', 'IMMUNOCOMPROMISED', 'OTHER']
for c in como:
    if c in df.columns:
        n1 = (df[c]==1).sum()
        total = df[c].notna().sum()
        print(f'  {c}: {n1}/{total} ({100*n1/total:.1f}% if present)')

# 6. Missingness of key clinical variables
print('\n--- MISSINGNESS (MEDIAN values for key biomarkers) ---')
key_labs = ['CREATININ_MEDIAN', 'GLUCOSE_MEDIAN', 'HEMOGLOBIN_MEDIAN',
            'LEUKOCYTES_MEDIAN', 'LINFOCITOS_MEDIAN', 'PLATELETS_MEDIAN',
            'PCR_MEDIAN', 'DIMER_MEDIAN', 'POTASSIUM_MEDIAN', 'SODIUM_MEDIAN',
            'TGO_MEDIAN', 'TGP_MEDIAN', 'UREA_MEDIAN', 'ALBUMIN_MEDIAN',
            'OXYGEN_SATURATION_MEDIAN']
miss_table = []
for c in key_labs:
    if c in df.columns:
        miss = df[c].isna().sum()
        pct = 100*miss/len(df)
        miss_table.append({'variable': c, 'missing': miss, 'pct_missing': round(pct,1)})
miss_df = pd.DataFrame(miss_table)
print(miss_df.to_string(index=False))

# 7. Save cleaned per-patient summary
print('\n--- SAVING FILES ---')

# Per-patient outcome file
patient_summary = df.groupby('PATIENT_VISIT_IDENTIFIER').agg({
    'ICU': 'max',
    'AGE_ABOVE65': 'first',
    'GENDER': 'first',
    'HTN': 'first',
    'IMMUNOCOMPROMISED': 'first'
}).reset_index()
patient_summary.to_csv(OUT_DIR / 'patient_summary.csv', index=False)
print(f'  Saved: {OUT_DIR}/patient_summary.csv ({len(patient_summary)} rows)')

# Missingness report
miss_df.to_csv(OUT_DIR / 'missingness_report.csv', index=False)
print(f'  Saved: {OUT_DIR}/missingness_report.csv')

# Column type summary
type_summary = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes.astype(str).values,
    'n_missing': df.isna().sum().values,
    'pct_missing': (100*df.isna().sum()/len(df)).round(1).values,
    'n_unique': df.nunique().values
})
type_summary.to_csv(OUT_DIR / 'all_columns_summary.csv', index=False)
print(f'  Saved: {OUT_DIR}/all_columns_summary.csv')

print('\n' + '='*70)
print('EDA COMPLETE. Check CSV files in eda_report/ folder.')
print('='*70)
