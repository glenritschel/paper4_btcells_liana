#!/usr/bin/env python3

import sys
import scanpy as sc

if len(sys.argv) != 2:
    print("Usage: python inspect_mito_genes.py <path_to_h5ad>")
    sys.exit(1)

path = sys.argv[1]
adata = sc.read_h5ad(path)

varnames = adata.var_names.astype(str)

mt = [g for g in varnames if g.startswith("MT-") or g.startswith("mt-")]

print(f"Total MT genes detected: {len(mt)}")
print("First 30 MT genes:")
for g in mt[:30]:
    print(g)

