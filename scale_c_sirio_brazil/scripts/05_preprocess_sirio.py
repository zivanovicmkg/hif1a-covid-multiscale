"""
===============================================================================
Scale C Preprocessing — Sirio-Libanes Hospital ICU Prediction Dataset
===============================================================================
Purpose:
    Prepare Sirio-Libanes data for ML training (Week 3-4 work).
    Critical design decisions:
    1. Use only WINDOW '0-2' (admission window) to prevent data leakage
       — patients already in ICU look very different from those not yet there.
    2. Select features that overlap with Marko's Kragujevac cohort (Scale D)
       for later external validation transfer.
    3. Use conservative missingness cutoff (<50%) for feature retention.

Input:
    /home/marko-b2/COVID_DATASETS/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx
Outputs (in /home/marko-b2/COVID_AI_Project/01_Data_Clinical_10k/sirio_processed/):
    - sirio_admission_window.csv       : raw 0-2h window, 385 patients
    - sirio_features_selected.csv      : features with <50% missing
    - sirio_ml_ready.csv               : after KNN imputation, ready for ML
    - sirio_feature_map.csv            : mapping to Marko's cohort variables
    - sirio_preprocessing_log.txt      : full log of decisions made

Author: Marko Živanović
Date: 2026-04-22
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

LOG_PATH = OUT_DIR / 'sirio_preprocessing_log.txt'
log_lines = []

def log(msg):
    """Print and log a message."""
    print(msg)
    log_lines.append(msg)

log('='*75)
log('SCALE C PREPROCESSING — SIRIO-LIBANES HOSPITAL ICU PREDICTION')
log(f'Run started: {datetime.now().isoformat()}')
log('='*75)

# ============================================================================
# STEP 1: Load and basic validation
# ============================================================================
log('\n[STEP 1] Loading data...')
df = pd.read_excel(INPUT)
log(f'  Raw shape: {df.shape[0]} rows x {df.shape[1]} cols')
log(f'  Unique patients: {df["PATIENT_VISIT_IDENTIFIER"].nunique()}')
log(f'  Windows present: {sorted(df["WINDOW"].unique().tolist())}')

# ============================================================================
# STEP 2: Select admission window only (critical — prevent leakage)
# ============================================================================
log('\n[STEP 2] Selecting admission window (0-2h) to prevent data leakage...')
log('  RATIONALE: Later windows contain post-deterioration data —')
log('  a patient who IS in ICU looks very different. Using the first window')
log('  simulates real clinical decision-making at hospital admission.')

df_adm = df[df['WINDOW'] == '0-2'].copy()
log(f'  Admission window shape: {df_adm.shape[0]} patients x {df_adm.shape[1]} cols')

# Compute patient-level ICU outcome (EVER went to ICU)
patient_ever_icu = df.groupby('PATIENT_VISIT_IDENTIFIER')['ICU'].max().reset_index()
patient_ever_icu.columns = ['PATIENT_VISIT_IDENTIFIER', 'ICU_EVER']

df_adm = df_adm.merge(patient_ever_icu, on='PATIENT_VISIT_IDENTIFIER', how='left')
log(f'  Patients who ever went to ICU: {(df_adm["ICU_EVER"]==1).sum()} '
    f'({100*(df_adm["ICU_EVER"]==1).mean():.1f}%)')
log(f'  Patients who never went to ICU: {(df_adm["ICU_EVER"]==0).sum()} '
    f'({100*(df_adm["ICU_EVER"]==0).mean():.1f}%)')

# Save admission window raw
df_adm.to_csv(OUT_DIR / 'sirio_admission_window.csv', index=False)
log(f'  Saved: sirio_admission_window.csv')

# ============================================================================
# STEP 3: Feature selection based on missingness
# ============================================================================
log('\n[STEP 3] Selecting features with <50% missingness at admission...')

# Exclude ID and target-related columns from feature set
exclude = ['PATIENT_VISIT_IDENTIFIER', 'WINDOW', 'ICU', 'ICU_EVER']
candidate_features = [c for c in df_adm.columns if c not in exclude]

miss_pct = (100 * df_adm[candidate_features].isna().sum() / len(df_adm)).round(1)
features_kept = miss_pct[miss_pct < 50].index.tolist()
features_dropped = miss_pct[miss_pct >= 50].index.tolist()

log(f'  Candidate features: {len(candidate_features)}')
log(f'  Kept (<50% missing): {len(features_kept)}')
log(f'  Dropped (>=50% missing): {len(features_dropped)}')

# ============================================================================
# STEP 4: Map Sirio variables to Marko's cohort variables
# ============================================================================
log('\n[STEP 4] Mapping Sirio features to Kragujevac cohort variables...')
log('  This is critical for later external validation (Scale C -> Scale D).')

# Mapping: Sirio column prefix -> Marko's variable name
# Covers variables that are measured in both cohorts
mapping = [
    # Demographics
    ('AGE_ABOVE65',        'age_65plus',           'binary age >= 65 (Marko: godine)'),
    ('AGE_PERCENTIL',      'age_percentile',       'age percentile (derived from Marko: godine)'),
    ('GENDER',             'sex',                  'Marko: pol'),
    ('HTN',                'htn',                  'Marko: HTA'),
    ('IMMUNOCOMPROMISED',  'immunocomp',           'Marko: maligna_Dg + HRI'),

    # Vital signs (Marko has SAT and pO2 at admission)
    ('OXYGEN_SATURATION',  'spo2',                 'Marko: SAT'),
    ('TEMPERATURE',        'temperature',          'Marko: not recorded explicitly'),
    ('HEART_RATE',         'heart_rate',           'Marko: not recorded'),
    ('RESPIRATORY_RATE',   'resp_rate',            'Marko: not recorded'),
    ('BLOODPRESSURE_SISTOLIC',  'sbp',             'Marko: TA implicit'),
    ('BLOODPRESSURE_DIASTOLIC', 'dbp',             'Marko: TA implicit'),

    # Laboratory — the most important for HIF1A story
    ('HEMOGLOBIN',         'hgb',                  'Marko: Hgb'),
    ('LEUKOCYTES',         'leukocytes',           'Marko: Le'),
    ('LINFOCITOS',         'lymphocytes',          'Marko: ly'),
    ('NEUTROPHILES',       'neutrophiles',         'derived Le - ly'),
    ('PLATELETS',          'platelets',            'Marko: Tr'),
    ('HEMATOCRITE',        'hematocrit',           'Marko: derived'),
    ('PCR',                'crp',                  'Marko: CRP (KEY HYPOXIA-INFLAMMATION MARKER)'),
    ('DIMER',              'd_dimer',              'Marko: D_dimer (KEY COAGULATION)'),
    ('CREATININ',          'creatinine',           'Marko: kreatinin'),
    ('UREA',               'urea',                 'Marko: urea'),
    ('GLUCOSE',            'glucose',              'Marko: glikemija'),
    ('ALBUMIN',            'albumin',              'Marko: alb'),
    ('POTASSIUM',          'potassium',            'Marko: K'),
    ('SODIUM',             'sodium',               'Marko: Na'),
    ('TGO',                'ast',                  'Marko: AST'),
    ('TGP',                'alt',                  'Marko: ALT'),
    ('INR',                'inr',                  'Marko: INR'),
    ('TTPA',               'aptt',                 'Marko: aPTT'),
    ('LACTATE',            'lactate',              'HYPOXIA MARKER (Marko: not measured)'),
    ('CALCIUM',            'calcium',              'Marko: not routinely'),
    ('BILLIRUBIN',         'bilirubin',            'Marko: billT'),
    ('GGT',                'ggt',                  'Marko: gGT'),
    ('BIC_VENOUS',         'bicarb_ven',           'Acid-base (not in Marko)'),
    ('BE_VENOUS',          'be_ven',               'Acid-base (not in Marko)'),
    ('PH_VENOUS',          'ph_ven',               'Acid-base (not in Marko)'),
    ('P02_VENOUS',         'po2_ven',              'Marko: pO2'),
    ('PC02_VENOUS',        'pco2_ven',             'Acid-base (not in Marko)'),
]

map_df = pd.DataFrame(mapping, columns=['sirio_prefix', 'marko_name', 'notes'])

# Check which mapped variables actually have data at admission
actually_present = []
for prefix, mname, notes in mapping:
    # Look for MEDIAN, MEAN, or exact match
    matching_cols = [c for c in features_kept if c.startswith(prefix)]
    if matching_cols:
        for c in matching_cols:
            actually_present.append({
                'sirio_prefix': prefix,
                'sirio_full_col': c,
                'marko_name': mname,
                'pct_missing': miss_pct[c],
                'notes': notes
            })

present_df = pd.DataFrame(actually_present)
log(f'  Mapped variables present in admission window: {len(present_df)}')
log(f'  Saved feature map: sirio_feature_map.csv')
map_df.to_csv(OUT_DIR / 'sirio_feature_map.csv', index=False)
present_df.to_csv(OUT_DIR / 'sirio_features_present.csv', index=False)

# ============================================================================
# STEP 5: Build selected feature dataframe
# ============================================================================
log('\n[STEP 5] Building selected feature dataframe...')

# Prefer MEDIAN values (most stable), fall back to MEAN
preferred_cols = []
for prefix, mname, notes in mapping:
    median_col = f'{prefix}_MEDIAN'
    mean_col = f'{prefix}_MEAN'
    single_col = prefix  # some variables don't have suffixes (AGE_ABOVE65, GENDER, HTN)
    
    if median_col in features_kept:
        preferred_cols.append((median_col, mname))
    elif mean_col in features_kept:
        preferred_cols.append((mean_col, mname))
    elif single_col in features_kept:
        preferred_cols.append((single_col, mname))

log(f'  Selected {len(preferred_cols)} mapped features for ML')

# Extract and rename
selected_data = df_adm[['PATIENT_VISIT_IDENTIFIER', 'ICU_EVER']].copy()
for sirio_col, marko_name in preferred_cols:
    selected_data[marko_name] = df_adm[sirio_col].values

log(f'  Selected DataFrame shape: {selected_data.shape}')

# Per-feature missingness after selection
log('\n  Per-feature missingness in selected data:')
for col in selected_data.columns[2:]:  # skip ID and target
    n_miss = selected_data[col].isna().sum()
    pct = 100*n_miss/len(selected_data)
    log(f'    {col:30s}  missing={n_miss:3d} ({pct:5.1f}%)')

selected_data.to_csv(OUT_DIR / 'sirio_features_selected.csv', index=False)
log(f'\n  Saved: sirio_features_selected.csv')

# ============================================================================
# STEP 6: KNN imputation for missing values
# ============================================================================
log('\n[STEP 6] Imputing missing values using KNN (k=5)...')
log('  RATIONALE: KNN preserves local data structure, better than mean for')
log('  correlated clinical variables. Patients are imputed based on their')
log('  most similar neighbors in the dataset.')

from sklearn.impute import KNNImputer

# Separate ID, target, and features
X_cols = [c for c in selected_data.columns if c not in ['PATIENT_VISIT_IDENTIFIER', 'ICU_EVER']]
X = selected_data[X_cols].values
ids = selected_data['PATIENT_VISIT_IDENTIFIER'].values
y = selected_data['ICU_EVER'].values

imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imp = imputer.fit_transform(X)

ml_ready = pd.DataFrame(X_imp, columns=X_cols)
ml_ready['PATIENT_VISIT_IDENTIFIER'] = ids
ml_ready['ICU_EVER'] = y.astype(int)

# Reorder columns: ID, target, features
col_order = ['PATIENT_VISIT_IDENTIFIER', 'ICU_EVER'] + X_cols
ml_ready = ml_ready[col_order]

ml_ready.to_csv(OUT_DIR / 'sirio_ml_ready.csv', index=False)
log(f'  Saved: sirio_ml_ready.csv ({ml_ready.shape[0]} patients, {len(X_cols)} features)')

# ============================================================================
# STEP 7: Summary statistics on final ML-ready dataset
# ============================================================================
log('\n[STEP 7] Final ML-ready dataset summary:')
log(f'  Total patients: {len(ml_ready)}')
log(f'  ICU EVER=1: {(ml_ready["ICU_EVER"]==1).sum()} ({100*(ml_ready["ICU_EVER"]==1).mean():.1f}%)')
log(f'  ICU EVER=0: {(ml_ready["ICU_EVER"]==0).sum()} ({100*(ml_ready["ICU_EVER"]==0).mean():.1f}%)')
log(f'  Features: {len(X_cols)}')
log('\n  Feature overlap with Marko cohort:')
high_priority = ['crp', 'd_dimer', 'spo2', 'po2_ven', 'hgb', 'leukocytes',
                 'lymphocytes', 'platelets', 'creatinine', 'urea', 'ast', 'alt', 'albumin']
present_hp = [f for f in high_priority if f in X_cols]
log(f'    High-priority overlapping features: {len(present_hp)}/{len(high_priority)}')
for f in high_priority:
    if f in X_cols:
        log(f'      [YES] {f}')
    else:
        log(f'      [NO ] {f} (missing >50% or not in Sirio)')

log('\n' + '='*75)
log(f'Preprocessing complete. Run ended: {datetime.now().isoformat()}')
log(f'Output folder: {OUT_DIR}')
log('='*75)

# Save log
with open(LOG_PATH, 'w') as f:
    f.write('\n'.join(log_lines))
print(f'\nLog saved to: {LOG_PATH}')
