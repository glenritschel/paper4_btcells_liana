import pandas as pd

df = pd.read_csv("results/liana/ssc_pbmc_gse210395.liana.tsv", sep="\t")

print("Columns available:")
print(df.columns.tolist())

targets = [
    "CD80", "CD86", "CD28", "CTLA4",
    "CD40", "CD40LG",
    "ICOS", "ICOSLG",
    "IL6", "IL6R", "IL6ST",
    "TGFB1", "TGFBR1", "TGFBR2",
    "IL1B", "IL1R1",
    "IL4", "IL4R",
    "IL13", "IL13RA1"
]

def contains(gene):
    return df[
        df["ligand_complex"].str.contains(gene, na=False) |
        df["receptor_complex"].str.contains(gene, na=False)
    ]

for g in targets:
    hits = contains(g)
    if len(hits) > 0:
        print("\n====", g, "====")
        print(
            hits[
                [
                    "source",
                    "target",
                    "ligand_complex",
                    "receptor_complex",
                    "lrscore",
                    "specificity_rank",
                    "magnitude_rank",
                ]
            ].head()
        )

