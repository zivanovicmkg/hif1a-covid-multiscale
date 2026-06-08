# Scale A — Data access

The Scale A analysis uses publicly available data from the **Mexican Secretariat of Health, Directorate General of Epidemiology (DGE) Open Data programme**.

## Source

- **URL**: https://www.gob.mx/salud/documentos/datos-abiertos-152127
- **Provider**: Secretaría de Salud, Dirección General de Epidemiología (DGE), Gobierno de México
- **License**: Términos de Libre Uso de los Datos Abiertos de la DGE (free use with attribution)

## Download procedure

1. Visit https://www.gob.mx/salud/documentos/datos-abiertos-152127
2. In the section **"Influenza, COVID-19 y otros virus respiratorios"** click **VER** next to **Base de Datos** to download the current weekly snapshot (ZIP containing CSV).
3. Also download **Diccionario de Datos** (data dictionary) from the same section — required to decode categorical columns (`CLASIFICACION_FINAL_COVID`, `TIPO_PACIENTE`, etc.).
4. If reproducing the original Scale A analysis (which used historical data covering the acute pandemic phase), use the **"Cierre Datos abiertos históricos"** links at the bottom of the page for the relevant year(s) — these are frozen archival snapshots that do not change.

## Inclusion criteria applied in `13_scale_a_mexico.py`

- `TIPO_PACIENTE == 2` (hospitalized)
- `CLASIFICACION_FINAL_COVID ∈ {1, 2, 3}` (PCR-confirmed SARS-CoV-2)
- Complete records for the variables used by the regression model

The selection yields **N = 35,268** patients in the original snapshot used in the manuscript.

## Citation

Secretaría de Salud, Dirección General de Epidemiología. *Datos abiertos: Influenza, COVID-19 y otros virus respiratorios* [Open data: Influenza, COVID-19 and other respiratory viruses] [Internet]. Mexico City: Gobierno de México; 2020 [accessed YYYY-MM-DD]. Available from: https://www.gob.mx/salud/documentos/datos-abiertos-152127

## Notes

- The DGE periodically rotates the live database: only the most recent ~18 months are exposed in the main download. Use **Bases Históricas / Cierre Datos abiertos históricos** links for older snapshots.
- The data dictionary changes occasionally — always download the dictionary alongside the data and confirm column codes (especially `CLASIFICACION_FINAL_COVID`, which underwent a code revision in 2022).
