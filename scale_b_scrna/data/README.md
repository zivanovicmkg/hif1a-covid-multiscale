# Scale B (scRNA) — Data access

Single-cell RNA-seq data for Scale B is publicly available from NCBI GEO.

## Source

- **GEO accession**: GSE234904
- **URL**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE234904
- **Publication**: Ito H, et al. *Neutrophil gene expression in COVID-19 patients with acute respiratory distress syndrome.* Front Immunol. 2025;16:1620745. doi:10.3389/fimmu.2025.1620745.
- **Original processing code**: https://github.com/amufaamo/covid-neutrophil-ards

## Download procedure

1. Visit the GEO accession page above.
2. Download `GSE234904_RAW.tar` (contains the per-sample 10x Genomics feature-barcode matrices).
3. Extract the tar archive into your local working tree (e.g. `data/GSE234904/unpacked/`).
4. Optional: also download `GSE234904_series_matrix.txt` for sample-level metadata.

After extraction, expected file pattern per sample:
```
GSM7476346_<barcode>_raw_feature.h5
GSM7476347_<barcode>_raw_feature.h5
...
```

The `17_scale_b_scRNA_phase1.py` script expects this layout. Edit the input path at the top of the script if your local layout differs.

## Cohort composition note

GSE234904 contains 5 COVID-19 ARDS patients + 6 healthy controls. The 6 healthy donors are **multiplexed in a single GSM** (GSM7476346) via TotalSeq-C hashtag antibody demultiplexing. The phase-1 script handles this demultiplexing.

## Hashtag demultiplexing

Hashtag-to-condition mapping is documented in the GSE234904 supplementary file `GSM7476347_Hashtag-condition.xlsx` (also retrievable from the original GitHub https://github.com/amufaamo/covid-neutrophil-ards). The script `17_scale_b_scRNA_phase1.py` reads this mapping at the top.

## Citation

Ito H, Ishikawa M, Yoshimura J, Liu Y, Sakakibara S, Sugihara F, Matsumoto H, Hirata H, Ogura H, Oda J, Okuzaki D. *Neutrophil gene expression in COVID-19 patients with acute respiratory distress syndrome.* Front Immunol. 2025;16:1620745. doi:10.3389/fimmu.2025.1620745.

Also cite the present multi-scale study when reusing this code.
