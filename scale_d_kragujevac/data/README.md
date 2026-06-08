# Scale D — Data access (Kragujevac clinical-genetic cohort)

## ⚠ Patient-level data is NOT available in this repository

The Scale D cohort consists of **93 patients hospitalized at the University Clinical Centre Kragujevac (UCC KG), Serbia**, with full clinical, biochemical, and genotypic data. Patient-level data **cannot be redistributed publicly** under the terms of the local ethics framework.

## Ethics framework

- **Approving body**: Ethics Committee, University Clinical Centre Kragujevac (UCC KG), Serbia
- **Approval decision**: No. **01/20-498**
- **Scope**: Prospective collection of clinical, biochemical, and genetic data from hospitalized COVID-19 patients; HIF1A genotyping of EDTA-anticoagulated whole-blood samples obtained as part of routine diagnostic workup; informed consent obtained from all patients or their legal representatives.

## How to request access

De-identified study data may be made available for **bona fide scientific re-analysis** upon reasonable request, subject to:

1. A written research proposal describing the intended re-analysis.
2. Approval by an independent review committee constituted by the study management group.
3. A signed data-access agreement (executed within 14 days of approval).
4. Verification that the proposed re-analysis is consistent with the original ethics approval (01/20-498) and the patients' informed consent.

**Contact**: Corresponding author of the published manuscript (see manuscript for contact details).

## What is available in this repository

- All Scale D **code** (`scripts/`) is published under the MIT license.
- Trained **models** (`models/`) contain coefficients and feature ordering only — no patient-level data.
- Aggregate **results** (`results/`) — metrics, coefficients, performance tables — are released under CC BY 4.0.

## What is NOT available

- Raw patient-level clinical / biochemical / genotypic data
- Patient identifiers of any kind
- Per-patient prediction outputs that could be linked to identifiable records

These are excluded by `.gitignore` at the repository root and are never committed.

## Citation

When reusing Scale D code or results, please cite the present multi-scale manuscript. Additionally, the genotyping platform was previously validated by our group at the same institution; please consider citing those works where methodologically appropriate:

- Matic S, et al. *IFNL3/4 polymorphisms as a two-edged sword: An association with COVID-19 outcome.* J Med Virol. 2023;95(2):e28506. doi:10.1002/jmv.28506.
- Matic S, et al. *Its all about IFN-λ4: Protective role of IFNL4 polymorphism against COVID-19-related pneumonia in females.* J Med Virol. 2023;95(11):e29152. doi:10.1002/jmv.29152.
- Matic S, et al. *ACE2 and TMPRSS2 genetic polymorphisms as potential predictors of COVID-19 severity and outcome in females.* Front Med (Lausanne). 2024;11:1493815. doi:10.3389/fmed.2024.1493815.
