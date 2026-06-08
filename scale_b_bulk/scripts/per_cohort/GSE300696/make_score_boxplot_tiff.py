import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

infile = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/hypoxia_inflammation_score.tsv"
out_png = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/plots/GSE300696_score_boxplot.png"
out_tiff = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/plots/GSE300696_score_boxplot.tiff"

df = pd.read_csv(infile, sep="\t")

groups = ["Hospitalized", "Convalescent"]
data = [df.loc[df["group"] == g, "HypoxiaInflammationScore"].dropna().values for g in groups]

fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.boxplot(data, labels=groups)
ax.set_ylabel("Hypoxia–Inflammation Score")
ax.set_title("GSE300696")
plt.xticks(rotation=15)
plt.tight_layout()

fig.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close(fig)

img = Image.open(out_png)
img.save(out_tiff, dpi=(600, 600), compression="tiff_lzw")

png_size = os.path.getsize(out_png) / (1024 * 1024)
tiff_size = os.path.getsize(out_tiff) / (1024 * 1024)

print("Saved PNG:", out_png, f"{png_size:.2f} MB")
print("Saved TIFF:", out_tiff, f"{tiff_size:.2f} MB")
