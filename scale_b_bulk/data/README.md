# Scale B (bulk) — Data access

All five bulk RNA-seq cohorts used in Scale B are publicly available from NCBI GEO. No raw counts or expression matrices are redistributed in this repository.

## Per-cohort accession and source publication

| GEO | Publication | DOI |
|---|---|---|
| GSE152075 | Lieberman NAP, et al. *In vivo antiviral host transcriptional response to SARS-CoV-2 by viral load, sex, and age.* PLOS Biol. 2020;18(9):e3000849. | 10.1371/journal.pbio.3000849 |
| GSE157103 | Overmyer KA, et al. *Large-scale multi-omic analysis of COVID-19 severity.* Cell Syst. 2021;12(1):23-40.e7. | 10.1016/j.cels.2020.10.003 |
| GSE171110 | Lévy Y, et al. *CD177, a specific marker of neutrophil activation, is associated with coronavirus disease 2019 severity and death.* iScience. 2021;24(7):102711. | 10.1016/j.isci.2021.102711 |
| GSE212861 | Rombauts A, et al. *Transcriptomic profiling of SARS-CoV-2 infected adult patients in mononuclear cells bulk RNA sequencing.* Biomedicines. 2023;11(5):1348. | 10.3390/biomedicines11051348 |
| GSE300696 | Ryu / An, et al. *(See manuscript for full citation.)* Commun Biol. 2025;8:1174. | (see manuscript) |

## Download procedure

For each accession:

1. Go to `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSEnnnnnn`
2. Download the **series matrix file** (clinical metadata):
   `GSEnnnnnn_series_matrix.txt.gz`
3. Download the **supplementary file** containing the raw counts / processed expression matrix (look under "Supplementary files" near the bottom of the GEO page).
4. Place the downloaded files under the corresponding cohort directory in your local working tree (e.g. `data/GSE152075/raw_data/`).

The processing scripts in `scripts/per_cohort/GSEnnnnnn/` expect the GEO-standard file naming convention. If file names differ, edit the path at the top of `parse_metadata_GSEnnnnnn.py`.

## Citation guidance

When reusing this code, please cite **both** the original cohort publications above **and** the present multi-scale study. Citations for each GEO accession are available on the corresponding GEO page under "Citations".

## Note on GSE171110 and GSE212861

These two accessions occasionally trigger NCBI bot-detection when fetched programmatically. If automated download fails, fetch the series matrix through a browser session. The processed files are otherwise identical.
