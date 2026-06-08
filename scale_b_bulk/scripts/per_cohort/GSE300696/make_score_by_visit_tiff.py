import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

infile = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/tables/hypoxia_inflammation_score.tsv"
out_png = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/plots/GSE300696_score_by_visit.png"
out_tiff = "/home/marko-b2/COVID_Transcriptomics_AI/06_scRNA_seq/GSE300696/plots/GSE300696_score_by_visit.tiff"

df = pd.read_csv(infile, sep="\t")

order = ["Visit1", "Visit2", "Visit3", "Visit4", "Visit5"]
data = [df.loc[df["visit"] == v, "HypoxiaInflammationScore"].dropna().values for v in order]

fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.boxplot(data, tick_labels=order)
ax.set_ylabel("Hypoxia–Inflammation Score")
ax.set_title("GSE300696")
ax.set_xlabel("Visit")
plt.tight_layout()

fig.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close(fig)

img = Image.open(out_png)
img.save(out_tiff, dpi=(600, 600), compression="tiff_lzw")

png_size = os.path.getsize(out_png) / (1024 * 1024)
tiff_size = os.path.getsize(out_tiff) / (1024 * 1024)

print("Saved PNG:", out_png, f"{png_size:.2f} MB")
print("Saved TIFF:", out_tiff, f"{tiff_size:.2f} MB")
