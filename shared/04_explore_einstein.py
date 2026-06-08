"""
Exploratory Data Analysis of Albert Einstein Hospital Brazil COVID-19 Dataset
==============================================================================
Input: /home/marko-b2/COVID_DATASETS/Einstein_Brazil/dataset.xlsx
Output: /home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/eda_report/einstein/

Known facts about this dataset (from literature):
- ~5644 patients, 111 features
- Data is z-score standardized (NOT original values)
- ~559-608 are COVID-positive (PCR confirmed)
- High missingness (~91% in raw version)
- Outcome: SARS-CoV-2 PCR result + ward/semi-ICU/ICU admission

Author: Marko Živanović
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

INPUT = Path('/home/marko-b2/COVID_DATASETS/Einstein_Brazil/dataset.xlsx')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/eda_report/einstein')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*70)
print('ALBERT EINSTEIN HOSPITAL (BRAZIL) — EXPLORATORY DATA ANALYSIS')
print('='*70)

# Load
df = pd.read_excel(INPUT)
print(f'\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns')

# 1. Column overview
print('\n--- COLUMN TYPES ---')
dtype_counts = df.dtypes.value_counts()
print(dtype_counts.to_string())

# 2. Overall missingness
total_cells = df.shape[0] * df.shape[1]
total_missing = df.isna().sum().sum()
print(f'\n--- OVERALL MISSINGNESS ---')
print(f'Total cells: {total_cells:,}')
print(f'Missing cells: {total_missing:,} ({100*total_missing/total_cells:.1f}%)')

# 3. Save full column summary
col_summary = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes.astype(str).values,
    'n_unique': df.nunique().values,
    'n_missing': df.isna().sum().values,
    'pct_missing': (100*df.isna().sum()/len(df)).round(1).values
})
col_summary = col_summary.sort_values('pct_missing')
col_summary.to_csv(OUT_DIR / 'column_summary.csv', index=False)
print(f'\nSaved: {OUT_DIR}/column_summary.csv')

# 4. Columns with < 50% missing (usable)
usable = col_summary[col_summary['pct_missing'] < 50]
print(f'\n--- USABLE COLUMNS (<50% missing): {len(usable)} of {len(col_summary)} ---')
print(usable.to_string(index=False))

# 5. Columns with < 30% missing (high quality)
high_quality = col_summary[col_summary['pct_missing'] < 30]
print(f'\n--- HIGH-QUALITY COLUMNS (<30% missing): {len(high_quality)} ---')
print(high_quality[['column', 'pct_missing']].to_string(index=False))

# 6. Find outcome/target columns
print('\n--- OUTCOME/TARGET COLUMNS ---')
outcome_kw = ['sars', 'covid', 'ward', 'icu', 'intens', 'semi', 'regular',
              'admission', 'result', 'exam']
outcome_cols = [c for c in df.columns if any(kw in c.lower() for kw in outcome_kw)]
for c in outcome_cols:
    vc = df[c].value_counts(dropna=False)
    print(f'\n{c}:')
    print(vc.to_string())

# 7. Look for HIF1A-pathway-relevant labs
print('\n--- HYPOXIA/INFLAMMATION RELEVANT VARIABLES ---')
bio_kw = ['lactat', 'ldh', 'dimer', 'crp', 'ferrit', 'il6', 'il-6',
          'proteina', 'troponin', 'hemoglobin', 'leukoc', 'linfoc',
          'neutrof', 'platelet', 'urea', 'creat', 'glucose', 'glucos',
          'ph', 'so2', 'sao2', 'sat', 'o2', 'po2', 'pco2', 'bicarb',
          'lymph', 'myelocyte', 'metamyelocyte']
bio_cols = [c for c in df.columns if any(kw in c.lower() for kw in bio_kw)]
bio_summary = col_summary[col_summary['column'].isin(bio_cols)].sort_values('pct_missing')
print(bio_summary.to_string(index=False))
bio_summary.to_csv(OUT_DIR / 'bio_relevant_cols.csv', index=False)

# 8. Save sample of the data (head)
df.head(10).to_csv(OUT_DIR / 'sample_head.csv', index=False)
print(f'\nSaved first 10 rows: {OUT_DIR}/sample_head.csv')

print('\n' + '='*70)
print('EDA COMPLETE. Next step: design preprocessing strategy')
print('based on usable columns and outcome distribution.')
print('='*70)
