"""
===============================================================================
Scale C Preprocessing v2 — Sirio-Libanes EARLY PREDICTION framework
===============================================================================
Scientific framework: Early prediction
    - INPUT window: 0-2h + 2-4h post-admission (early clinical data)
    - TARGET: Subsequent ICU admission (in 4-6h, 6-12h, or ABOVE_12h)
    - EXCLUSIONS: Patients already in ICU at input window (prevents leakage)

Rationale:
    A clinician making an admission decision has ~first few hours of data.
    A useful model predicts what happens LATER, not what is currently happening.

Input:
    /home/marko-b2/COVID_DATASETS/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx
Outputs (in .../01_Data_Clinical_10k/sirio_processed/):
    - sirio_early_raw.csv              : early window data, before cleaning
    - sirio_early_features.csv         : selected features, before imputation
    - sirio_early_ml_ready.csv         : FINAL — imputed, ML-ready
    - sirio_v2_feature_map.csv         : mapping to Marko's cohort
    - sirio_v2_log.txt                 : full decision log

Author: Marko Živanović
Date: 2026-04-22 (v2)
===============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
INPUT = Path('/home/marko-b2/COVID_DATASETS/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
OUT_DIR = Path('/home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/sirio_processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / 'sirio_v2_log.txt'

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log('='*75)
log('SCALE C PREPROCESSING v2 — EARLY PREDICTION FRAMEWORK')
log(f'Started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# STEP 1: Load
# ============================================================================
log('\n[STEP 1] Loading data...')
df = pd.read_excel(INPUT)
log(f'  Loaded: {df.shape[0]} rows, {df.shape[1]} cols, {df["PATIENT_VISIT_IDENTIFIER"].nunique()} patients')

# ============================================================================
# STEP 2: Define input and outcome windows
# ============================================================================
log('\n[STEP 2] Defining early-prediction framework...')
INPUT_WINDOWS = ['0-2', '2-4']
OUTCOME_WINDOWS = ['4-6', '6-12', 'ABOVE_12']
log(f'  INPUT windows (clinical data used as predictors): {INPUT_WINDOWS}')
log(f'  OUTCOME windows (ICU label checked here): {OUTCOME_WINDOWS}')

# ============================================================================
# STEP 3: Identify exclusions — patients already in ICU at input window
# ============================================================================
log('\n[STEP 3] Identifying patients to exclude (ICU=1 in input window)...')

early_icu = df[df['WINDOW'].isin(INPUT_WINDOWS)].groupby('PATIENT_VISIT_IDENTIFIER')['ICU'].max()
exclude_ids = early_icu[early_icu == 1].index.tolist()

log(f'  Patients with ICU=1 in 0-2h or 2-4h window: {len(exclude_ids)}')
log(f'  These are excluded — we need patients NOT yet in ICU at input time.')

all_patients = df['PATIENT_VISIT_IDENTIFIER'].unique()
kept_ids = [p for p in all_patients if p not in exclude_ids]
log(f'  Kept patients: {len(kept_ids)} / {len(all_patients)}')

# ============================================================================
# STEP 4: Extract input-window data (aggregate across 0-2h + 2-4h)
# ============================================================================
log('\n[STEP 4] Aggregating input-window data per patient...')
log('  For each patient, take FIRST non-null value across 0-2h and 2-4h')
log('  (this preserves earliest measurement when available)')

df_input = df[
    (df['WINDOW'].isin(INPUT_WINDOWS)) &
    (df['PATIENT_VISIT_IDENTIFIER'].isin(kept_ids))
].copy()

# Order by window to get 0-2 first, then 2-4
window_order = {w: i for i, w in enumerate(INPUT_WINDOWS)}
df_input['window_order'] = df_input['WINDOW'].map(window_order)
df_input = df_input.sort_values(['PATIENT_VISIT_IDENTIFIER', 'window_order'])

# Aggregate: first non-null per patient
df_agg = df_input.groupby('PATIENT_VISIT_IDENTIFIER').first().reset_index()
df_agg = df_agg.drop(columns=['WINDOW', 'window_order', 'ICU'])  # remove; will add outcome separately

log(f'  Aggregated shape: {df_agg.shape}')

# ============================================================================
# STEP 5: Attach outcome (ICU in later windows)
# ============================================================================
log('\n[STEP 5] Computing outcome: ICU admission in later windows...')

df_outcome = df[
    (df['WINDOW'].isin(OUTCOME_WINDOWS)) &
    (df['PATIENT_VISIT_IDENTIFIER'].isin(kept_ids))
]

icu_later = df_outcome.groupby('PATIENT_VISIT_IDENTIFIER')['ICU'].max().reset_index()
icu_later.columns = ['PATIENT_VISIT_IDENTIFIER', 'ICU_LATER']

df_agg = df_agg.merge(icu_later, on='PATIENT_VISIT_IDENTIFIER', how='left')
df_agg['ICU_LATER'] = df_agg['ICU_LATER'].fillna(0).astype(int)

n_pos = (df_agg['ICU_LATER'] == 1).sum()
n_neg = (df_agg['ICU_LATER'] == 0).sum()
log(f'  Outcome distribution:')
log(f'    ICU_LATER=1 (went to ICU later): {n_pos} ({100*n_pos/len(df_agg):.1f}%)')
log(f'    ICU_LATER=0 (never went to ICU): {n_neg} ({100*n_neg/len(df_agg):.1f}%)')

df_agg.to_csv(OUT_DIR / 'sirio_early_raw.csv', index=False)
log(f'  Saved: sirio_early_raw.csv')

# ============================================================================
# STEP 6: Convert AGE_PERCENTIL to numeric
# ============================================================================
log('\n[STEP 6] Converting AGE_PERCENTIL to numeric...')

def parse_age_pct(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if 'Above' in s:
        return 95.0
    # Extract leading digits
    import re
    m = re.match(r'(\d+)', s)
    if m:
        return float(m.group(1))
    return np.nan

df_agg['AGE_PERCENTIL_NUM'] = df_agg['AGE_PERCENTIL'].apply(parse_age_pct)
log(f'  AGE_PERCENTIL_NUM range: {df_agg["AGE_PERCENTIL_NUM"].min()} to {df_agg["AGE_PERCENTIL_NUM"].max()}')
log(f'  Distribution:')
for bin_val in sorted(df_agg['AGE_PERCENTIL_NUM'].unique()):
    n = (df_agg['AGE_PERCENTIL_NUM'] == bin_val).sum()
    log(f'    {bin_val:5.1f}: {n} patients')

# ============================================================================
# STEP 7: Feature mapping to Marko's cohort
# ============================================================================
log('\n[STEP 7] Building Sirio -> Kragujevac feature map...')

mapping = [
    # Demographics
    ('AGE_ABOVE65',           'age_65plus',     'binary age>=65'),
    ('AGE_PERCENTIL_NUM',     'age_percentile', 'age percentile numeric'),
    ('GENDER',                'sex',            'Marko: pol'),
    ('HTN',                   'htn',            'Marko: HTA'),
    ('IMMUNOCOMPROMISED',     'immunocomp',     'Marko: HRI + malign'),

    # Vital signs
    ('OXYGEN_SATURATION_MEDIAN',       'spo2',           'Marko: SAT'),
    ('RESPIRATORY_RATE_MEDIAN',        'resp_rate',      'vital sign'),
    ('HEART_RATE_MEDIAN',              'heart_rate',     'vital sign'),
    ('BLOODPRESSURE_SISTOLIC_MEDIAN',  'sbp',            'Marko: TA'),
    ('BLOODPRESSURE_DIASTOLIC_MEDIAN', 'dbp',            'Marko: TA'),
    ('TEMPERATURE_MEDIAN',             'temperature',    'vital sign'),

    # Labs — KEY for HIF1A story
    ('HEMOGLOBIN_MEDIAN',   'hgb',           'Marko: Hgb'),
    ('LEUKOCYTES_MEDIAN',   'leukocytes',    'Marko: Le'),
    ('LINFOCITOS_MEDIAN',   'lymphocytes',   'Marko: ly'),
    ('NEUTROPHILES_MEDIAN', 'neutrophiles',  'derived'),
    ('PLATELETS_MEDIAN',    'platelets',     'Marko: Tr'),
    ('PCR_MEDIAN',          'crp',           'Marko: CRP — KEY INFLAMMATION'),
    ('DIMER_MEDIAN',        'd_dimer',       'Marko: D_dimer — KEY COAG'),
    ('CREATININ_MEDIAN',    'creatinine',    'Marko: kreatinin'),
    ('UREA_MEDIAN',         'urea',          'Marko: urea'),
    ('GLUCOSE_MEDIAN',      'glucose',       'Marko: glikemija'),
    ('ALBUMIN_MEDIAN',      'albumin',       'Marko: alb'),
    ('POTASSIUM_MEDIAN',    'potassium',     'Marko: K'),
    ('SODIUM_MEDIAN',       'sodium',        'Marko: Na'),
    ('TGO_MEDIAN',          'ast',           'Marko: AST'),
    ('TGP_MEDIAN',          'alt',           'Marko: ALT'),
    ('INR_MEDIAN',          'inr',           'Marko: INR'),
    ('TTPA_MEDIAN',         'aptt',          'Marko: aPTT'),
    ('LACTATE_MEDIAN',      'lactate',       'HYPOXIA marker'),
    ('BILLIRUBIN_MEDIAN',   'bilirubin',     'Marko: billT'),
]

# Select and rename
feature_rows = []
selected_data = df_agg[['PATIENT_VISIT_IDENTIFIER', 'ICU_LATER']].copy()

for sirio_col, marko_name, notes in mapping:
    if sirio_col in df_agg.columns:
        selected_data[marko_name] = df_agg[sirio_col].values
        n_present = df_agg[sirio_col].notna().sum()
        pct = 100 * n_present / len(df_agg)
        feature_rows.append({
            'sirio_col': sirio_col,
            'marko_name': marko_name,
            'n_present': n_present,
            'pct_present': round(pct, 1),
            'notes': notes
        })

fmap_df = pd.DataFrame(feature_rows).sort_values('pct_present', ascending=False)
fmap_df.to_csv(OUT_DIR / 'sirio_v2_feature_map.csv', index=False)

log('\n  Feature availability in early window:')
for _, r in fmap_df.iterrows():
    mark = '  [KEEP]' if r['pct_present'] >= 50 else '  [DROP]'
    log(f'  {mark} {r["marko_name"]:18s} {r["pct_present"]:5.1f}% present  ({r["sirio_col"]})')

# Keep only features with >=50% present
keep_features = fmap_df[fmap_df['pct_present'] >= 50]['marko_name'].tolist()
log(f'\n  Features kept (>=50% present): {len(keep_features)}')

final_cols = ['PATIENT_VISIT_IDENTIFIER', 'ICU_LATER'] + keep_features
features_df = selected_data[final_cols].copy()
features_df.to_csv(OUT_DIR / 'sirio_early_features.csv', index=False)
log(f'  Saved: sirio_early_features.csv')

# ============================================================================
# STEP 8: KNN imputation
# ============================================================================
log('\n[STEP 8] KNN imputation (k=5, distance-weighted)...')
from sklearn.impute import KNNImputer

X_cols = [c for c in features_df.columns if c not in ['PATIENT_VISIT_IDENTIFIER', 'ICU_LATER']]

# Safety check — all should be numeric now
non_numeric = []
for c in X_cols:
    try:
        features_df[c].astype(float)
    except (ValueError, TypeError):
        non_numeric.append(c)

if non_numeric:
    log(f'  WARNING: non-numeric columns detected: {non_numeric}')
    log(f'  These will be dropped.')
    X_cols = [c for c in X_cols if c not in non_numeric]

X = features_df[X_cols].astype(float).values
ids = features_df['PATIENT_VISIT_IDENTIFIER'].values
y = features_df['ICU_LATER'].values

imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imp = imputer.fit_transform(X)

ml_ready = pd.DataFrame(X_imp, columns=X_cols)
ml_ready.insert(0, 'PATIENT_VISIT_IDENTIFIER', ids)
ml_ready.insert(1, 'ICU_LATER', y.astype(int))

ml_ready.to_csv(OUT_DIR / 'sirio_early_ml_ready.csv', index=False)
log(f'  Saved: sirio_early_ml_ready.csv — {ml_ready.shape[0]} patients x {len(X_cols)} features')

# ============================================================================
# STEP 9: Final summary
# ============================================================================
log('\n[STEP 9] Final summary:')
log(f'  Total patients in ML-ready: {len(ml_ready)}')
log(f'  Outcome ICU_LATER=1: {(ml_ready["ICU_LATER"]==1).sum()} ({100*(ml_ready["ICU_LATER"]==1).mean():.1f}%)')
log(f'  Outcome ICU_LATER=0: {(ml_ready["ICU_LATER"]==0).sum()} ({100*(ml_ready["ICU_LATER"]==0).mean():.1f}%)')
log(f'  Features: {len(X_cols)}')
log(f'\n  Feature list:')
for c in X_cols:
    log(f'    - {c}')

log(f'\n  Outcome-to-features ratio: {len(ml_ready)}/{len(X_cols)} = {len(ml_ready)/len(X_cols):.1f} patients per feature')
log('  Rule-of-thumb minimum for stable ML: 10-20 patients per feature')

log('\n' + '='*75)
log(f'Done: {datetime.now().isoformat()}')
log('='*75)

with open(LOG_PATH, 'w') as f:
    f.write('\n'.join(log_lines))
print(f'\nLog saved to: {LOG_PATH}')
