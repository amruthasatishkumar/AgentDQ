"""Build the OULAD early-warning table for the validator paper.

One row per student-presentation, with only what is known by day CUTOFF of the
course (demographics, registration, assessments due by then, VLE clicks by then).
Label y = 1 if the final result is Fail or Withdrawn, 0 for Pass or Distinction.
Students who unregistered on or before the cutoff are dropped: they are already
withdrawn, so predicting them is trivial.

Temporal split by presentation: train = 2013B, 2013J; val (probe) = 2014B; test = 2014J.

Run:
    python3 -I build_oulad.py --zip data/oulad/oulad.zip --out data/oulad/oulad_day28.parquet
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

LABEL = "y"
CUTOFF = 28
CATEGORICAL = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band",
               "disability"]
NUMERIC = ["num_of_prev_attempts", "studied_credits", "date_registration",
           "n_assess_due", "n_submitted", "n_missed", "mean_score", "min_score",
           "mean_days_early", "clicks_total", "active_days", "n_resources_used"]
ENGAGEMENT = ["n_assess_due", "n_submitted", "n_missed", "mean_score", "min_score",
              "mean_days_early", "clicks_total", "active_days", "n_resources_used"]
SPLITS = {"2013B": "train", "2013J": "train", "2014B": "val", "2014J": "test"}
KEY = ["code_module", "code_presentation", "id_student"]


def _read(z: zipfile.ZipFile, name: str, **kw) -> pd.DataFrame:
    with z.open(name) as f:
        return pd.read_csv(f, na_values=["?"], **kw)


def build(zip_path: Path, cutoff: int = CUTOFF) -> pd.DataFrame:
    z = zipfile.ZipFile(zip_path)
    info = _read(z, "studentInfo.csv")
    reg = _read(z, "studentRegistration.csv")
    assess = _read(z, "assessments.csv")
    sa = _read(z, "studentAssessment.csv")

    df = info.merge(reg, on=KEY, how="left")
    df = df[~(df.date_unregistration <= cutoff)].copy()
    df[LABEL] = df.final_result.isin(["Fail", "Withdrawn"]).astype(int)

    # assessments due by the cutoff (exams have no date and are never early)
    due = assess[(assess.date <= cutoff) & (assess.assessment_type != "Exam")]
    n_due = due.groupby(["code_module", "code_presentation"]).size().rename("n_assess_due")
    df = df.merge(n_due.reset_index(), on=["code_module", "code_presentation"], how="left")
    df["n_assess_due"] = df.n_assess_due.fillna(0)

    sub = sa.merge(due[["id_assessment", "code_module", "code_presentation", "date"]],
                   on="id_assessment")
    sub = sub[sub.date_submitted <= cutoff]
    sub["days_early"] = sub.date - sub.date_submitted
    agg = sub.groupby(KEY).agg(n_submitted=("score", "size"), mean_score=("score", "mean"),
                               min_score=("score", "min"), mean_days_early=("days_early", "mean"))
    df = df.merge(agg.reset_index(), on=KEY, how="left")
    df["n_submitted"] = df.n_submitted.fillna(0)
    df["n_missed"] = (df.n_assess_due - df.n_submitted).clip(lower=0)

    # VLE clicks up to the cutoff, streamed (the file is ~450 MB)
    parts = []
    with z.open("studentVle.csv") as f:
        for ch in pd.read_csv(f, chunksize=2_000_000):
            parts.append(ch[ch.date <= cutoff])
    early = pd.concat(parts)
    vle = early.groupby(KEY).agg(clicks_total=("sum_click", "sum"), active_days=("date", "nunique"),
                                 n_resources_used=("id_site", "nunique"))
    df = df.merge(vle.reset_index(), on=KEY, how="left")
    for c in ("clicks_total", "active_days", "n_resources_used"):
        df[c] = df[c].fillna(0)

    df["split"] = df.code_presentation.map(SPLITS)
    df["row_id"] = df.code_module + "_" + df.code_presentation + "_" + df.id_student.astype(str)
    cols = ["row_id", "split", LABEL] + CATEGORICAL + NUMERIC
    out = df[cols].copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype("string")
    for c in NUMERIC:
        out[c] = out[c].astype(float)
    return out.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", type=int, default=CUTOFF)
    a = ap.parse_args()
    df = build(a.zip, a.cutoff)
    df.to_parquet(a.out, index=False)
    print(df.groupby("split")[LABEL].agg(["size", "mean"]))
    print(df[NUMERIC].describe().T[["mean", "50%", "max"]].round(2))
    print("missing:", df.isna().mean().round(3)[lambda s: s > 0].to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
