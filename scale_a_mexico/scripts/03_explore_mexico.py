"""
Exploratory Data Analysis of Mexico COVID-19 Open Government Dataset
=====================================================================
Input: /home/marko-b2/COVID_DATASETS/mexico_covid19.csv
Output: /home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/eda_report/mexico/

Goal: Understand what we have in the Mexico dataset — cohort size, outcome balance,
      comorbidities, and whether variables are useful for severity prediction.
Author: Marko Živanović
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

INPUT = Path('/home/marko-b2/COVID_DATASETS/mexico_covid19.csv')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/eda_report/mexico')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*70)
print('MEXICO COVID-19 OPEN DATA — EXPLORATORY DATA ANALYSIS')
print('='*70)

# Load — Mexico dataset is known to be large (263K+ rows), load efficiently
# First, check encoding and separator by peeking at file
print('\n--- FILE PEEK ---')
with open(INPUT, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        print(f'Line {i}: {line[:200]}')
        if i >= 2:
            break

# Try reading
print('\n--- LOADING FULL FILE ---')
try:
    df = pd.read_csv(INPUT, low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(INPUT, low_memory=False, encoding='latin-1')

print(f'Shape: {df.shape[0]:,} rows x {df.shape[1]} columns')

# Columns
print(f'\n--- COLUMNS ({len(df.columns)}) ---')
for i, col in enumerate(df.columns):
    dtype = str(df[col].dtype)
    n_unique = df[col].nunique()
    n_missing = df[col].isna().sum()
    pct_missing = 100*n_missing/len(df)
    print(f'  [{i:2d}] {col:40s}  dtype={dtype:10s}  unique={n_unique:6d}  missing={pct_missing:5.1f}%')

# Save column summary
col_summary = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes.astype(str).values,
    'n_unique': df.nunique().values,
    'n_missing': df.isna().sum().values,
    'pct_missing': (100*df.isna().sum()/len(df)).round(1).values
})
col_summary.to_csv(OUT_DIR / 'column_summary.csv', index=False)
print(f'\nSaved column summary: {OUT_DIR}/column_summary.csv')

# Look for outcome-relevant columns
print('\n--- LOOKING FOR OUTCOME-RELEVANT COLUMNS ---')
keywords = ['death', 'muert', 'fall', 'icu', 'uci', 'intub', 'hosp', 'ingreso',
            'tipo_pac', 'fecha_def', 'outcome', 'severity', 'grav', 'neumonia',
            'pneumonia']
candidates = [c for c in df.columns if any(kw in c.lower() for kw in keywords)]
print(f'Outcome candidates: {candidates}')

# Show first 5 values for each candidate
for c in candidates:
    vc = df[c].value_counts(dropna=False).head(10)
    print(f'\n{c}:')
    print(vc.to_string())

# Look for comorbidity columns
print('\n--- LOOKING FOR COMORBIDITY COLUMNS ---')
como_keywords = ['diab', 'hiperten', 'htn', 'obesi', 'asma', 'epoc', 'copd',
                 'cardio', 'renal', 'inmuno', 'immuno', 'tabaq', 'embarazo',
                 'preg', 'vih', 'hiv']
como_candidates = [c for c in df.columns if any(kw in c.lower() for kw in como_keywords)]
print(f'Comorbidity candidates: {como_candidates}')

# Look for demographics
print('\n--- LOOKING FOR DEMOGRAPHIC COLUMNS ---')
demo_keywords = ['edad', 'age', 'sexo', 'sex', 'gender', 'entidad', 'state',
                 'municipio', 'nacion']
demo_candidates = [c for c in df.columns if any(kw in c.lower() for kw in demo_keywords)]
print(f'Demographic candidates: {demo_candidates}')

# Look for any lab values (unlikely but check)
print('\n--- LOOKING FOR LAB VALUE COLUMNS (expected: none) ---')
lab_keywords = ['crp', 'dimer', 'ferrit', 'ldh', 'linfo', 'leuko', 'hemo',
                'trombo', 'platelet', 'creat', 'urea', 'ast', 'alt']
lab_candidates = [c for c in df.columns if any(kw in c.lower() for kw in lab_keywords)]
print(f'Lab value candidates: {lab_candidates}')
if not lab_candidates:
    print('  (As expected: Mexico dataset has NO laboratory values)')

print('\n' + '='*70)
print('EDA COMPLETE.')
print('='*70)
