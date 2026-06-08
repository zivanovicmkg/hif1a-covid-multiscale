import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--counts")
parser.add_argument("--metadata")
parser.add_argument("--genes")
parser.add_argument("--out")
args = parser.parse_args()

print("Loading counts...")
counts = pd.read_csv(args.counts, sep="\t")

print("Loading metadata...")
meta = pd.read_csv(args.metadata, sep="\t")

print("Loading gene panel...")
genes = [g.strip() for g in open(args.genes)]

print("Filtering genes...")
counts = counts[counts.iloc[:,0].isin(genes)]

print("Transposing...")
counts = counts.set_index(counts.columns[0]).T
counts.index.name = "sample"
counts.reset_index(inplace=True)

print("Merging metadata...")
merged = counts.merge(meta, on="sample")

print("Saving...")
merged.to_csv(args.out, sep="\t", index=False)

print("Done:", args.out)
