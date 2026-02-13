#!/usr/bin/env python3
"""
OLS regression for delta_rank vs expression changes.
Outputs JSON summary + TSV of coefficients (+ standardized betas).

Example:
  python scripts/regress_delta_rank.py \
    --input results/amplification_gse210395_blow_vs_bhigh/lr_amplification_gse210395_blow_vs_bhigh.tsv \
    --y delta_rank \
    --x lig_log2fc \
    --controls rec_log2fc \
    --outdir results/regression_gse210395
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


def fit_ols(df: pd.DataFrame, y: str, x: List[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = df[x].copy()
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(df[y], X, missing="drop")
    return model.fit()


def summarize_fit(fit, feature_names: List[str]) -> Dict[str, Any]:
    out = {
        "nobs": int(fit.nobs),
        "df_model": float(fit.df_model),
        "df_resid": float(fit.df_resid),
        "r2": float(fit.rsquared),
        "r2_adj": float(fit.rsquared_adj),
        "fvalue": None if fit.fvalue is None else float(fit.fvalue),
        "f_pvalue": None if fit.f_pvalue is None else float(fit.f_pvalue),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "params": {},
    }

    for name in ["const"] + feature_names:
        if name not in fit.params.index:
            continue
        out["params"][name] = {
            "coef": float(fit.params[name]),
            "se": float(fit.bse[name]),
            "t": float(fit.tvalues[name]),
            "p": float(fit.pvalues[name]),
            "ci_low": float(fit.conf_int().loc[name, 0]),
            "ci_high": float(fit.conf_int().loc[name, 1]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TSV with delta_rank and log2fc columns")
    ap.add_argument("--y", required=True, help="Outcome column, e.g. delta_rank")
    ap.add_argument("--x", required=True, nargs="+", help="Predictor(s), e.g. lig_log2fc")
    ap.add_argument("--controls", default=[], nargs="*", help="Optional controls added to model")
    ap.add_argument("--dropna", action="store_true", help="Drop rows with any NA in model columns")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="regression")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")
    cols = [args.y] + list(args.x) + list(args.controls)

    for c in cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    if args.dropna:
        work = work.dropna(axis=0, how="any", subset=cols)

    # Handle zero-variance controls explicitly (your rec_log2fc case)
    var = work[list(args.x) + list(args.controls)].var(ddof=1)
    zero_var = var[var == 0].index.tolist()
    kept_controls = [c for c in args.controls if c not in zero_var]
    kept_x = [c for c in args.x if c not in zero_var]
    dropped = zero_var

    features = kept_x + kept_controls
    if len(features) == 0:
        raise SystemExit("No non-constant predictors remain after filtering zero-variance columns.")

    # Unstandardized fit
    fit = fit_ols(work, args.y, features)
    summary = summarize_fit(fit, features)

    # Standardized beta fit (z-scored y and predictors; intercept still present but not used)
    zdf = work[[args.y] + features].copy()
    zdf[args.y] = zscore(zdf[args.y])
    for c in features:
        zdf[c] = zscore(zdf[c])

    zfit = fit_ols(zdf, args.y, features)
    zparams = {}
    for name in ["const"] + features:
        if name in zfit.params.index:
            zparams[name] = float(zfit.params[name])

    # Write coefficients table
    rows = []
    for name in ["const"] + features:
        if name not in fit.params.index:
            continue
        rows.append({
            "term": name,
            "coef": float(fit.params[name]),
            "se": float(fit.bse[name]),
            "t": float(fit.tvalues[name]),
            "p": float(fit.pvalues[name]),
            "ci_low": float(fit.conf_int().loc[name, 0]),
            "ci_high": float(fit.conf_int().loc[name, 1]),
            "beta_std": None if name not in zparams else zparams[name],
        })
    coef_df = pd.DataFrame(rows)

    # Save outputs
    prefix = args.prefix
    coef_path = outdir / f"{prefix}.coefs.tsv"
    json_path = outdir / f"{prefix}.summary.json"

    coef_df.to_csv(coef_path, sep="\t", index=False)

    summary["standardized_betas"] = zparams
    summary["dropped_zero_variance_predictors"] = dropped
    summary["features_used"] = features
    summary["y"] = args.y
    summary["x"] = args.x
    summary["controls_requested"] = args.controls
    summary["controls_used"] = kept_controls

    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote: {coef_path}")
    print(f"Wrote: {json_path}")
    if dropped:
        print(f"Dropped zero-variance predictors: {dropped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

