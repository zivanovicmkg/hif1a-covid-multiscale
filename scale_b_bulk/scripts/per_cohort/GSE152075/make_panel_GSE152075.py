import pandas as pd

meta = pd.read_csv("metadata_GSE152075.tsv", sep="\t")
genes = ["HIF1A","VEGFA","SLC2A1","LDHA","PDK1","IL6","TNF","CXCL8","STAT3","MMP9"]

with open("GSE152075_raw_counts_GEO.txt") as f:
    header_samples = f.readline().strip().split()

rows = []
for gene in genes:
    rows.append({"gene_symbol": gene})

gene_to_row = {r["gene_symbol"]: r for r in rows}

with open("GSE152075_raw_counts_GEO.txt") as f:
    next(f)  # preskoči header
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        gene = parts[0]
        if gene in gene_to_row:
            values = parts[1:]
            if len(values) != len(header_samples):
                print(f"WARNING: {gene} ima {len(values)} vrednosti, a header ima {len(header_samples)}")
                continue
            for s, v in zip(header_samples, values):
                gene_to_row[gene][s] = float(v)

panel = pd.DataFrame(rows)

matched = panel["gene_symbol"].tolist()

panel_t = panel.set_index("gene_symbol").T
panel_t.index.name = "expr_sample"
panel_t.reset_index(inplace=True)

merged = pd.merge(meta, panel_t, on="expr_sample")

out = "hypoxia_inflammation_panel_collapsed_GSE152075.tsv"
merged.to_csv(out, sep="\t", index=False)

print("Saved:", out)
print("Shape:", merged.shape)
print("\nMatched genes:", matched)
print("\nHead:")
print(merged.head())
