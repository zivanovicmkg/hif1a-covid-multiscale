# Scale C — Data access (Sírio-Libanês, Kaggle)

The Sírio-Libanês COVID-19 dataset is publicly available on Kaggle.

## Source

- **Hosting**: Kaggle
- **URL**: https://www.kaggle.com/datasets/S%C3%ADrio-Libanes/covid19
- **Provider**: Hospital Sírio-Libanês, São Paulo, Brazil
- **License**: CC0 1.0 Universal (Public Domain Dedication) per the Kaggle dataset metadata

## Download procedure

1. Create a free Kaggle account if you do not have one: https://www.kaggle.com/account/login
2. Navigate to the dataset page above.
3. Click **Download** (top-right of the dataset page). You will receive a ZIP archive containing the Excel file `Kaggle_Sirio_Libanes_data.xlsx` (filename may vary).
4. Place the file in your local working tree (e.g. `data/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx`).

Programmatic alternative using the Kaggle API:

```bash
pip install kaggle
# After configuring ~/.kaggle/kaggle.json with your API token:
kaggle datasets download -d S%C3%ADrio-Libanes/covid19 -p data/
unzip data/covid19.zip -d data/
```

## Citation

Hospital Sírio-Libanês. *COVID-19 Clinical Data to assess diagnosis* [Dataset]. Kaggle. Available from: https://www.kaggle.com/datasets/S%C3%ADrio-Libanes/covid19

Also cite the present multi-scale study when reusing this code.

## Important note about the dataset

The Kaggle dataset is anonymized per the dataset provider. Patient ages are bucketed into deciles ("AGE_PERCENTIL"); admission timestamps are pseudonymized into 2-hour relative windows. These design choices restrict some downstream analyses and are taken into account by `06_preprocess_sirio_v2.py`.
