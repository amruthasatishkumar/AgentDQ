# Experiments: consumer-aware validation

Scripts behind the paper *Is This Table Clean for You? Consumer-Aware Data Validation for Predictive Models and LLM Agents*. They use the released `agentdq` package; no data is stored here.

| Script | What it does |
|---|---|
| `run_experiments.py` | Olist negative-review task: XGBoost tree and a TF-IDF text model, RQ1–RQ3 |
| `build_oulad.py` | Builds the OULAD day-28 early-warning table (fail or withdraw vs pass) |
| `run_oulad.py` | OULAD: XGBoost tree and an engagement-only logistic regression, RQ1–RQ3 |
| `run_oulad_agent.py` | Adds a gpt-4o-mini agent on OULAD (Azure OpenAI), RQ1 for all three consumers |

All four share the batch, scoring and evaluation code in `run_experiments.py`: 0–3 corruption families per batch at severity 0–0.4, ground truth is the paired clean-minus-corrupted AUROC, and Spearman ρ is reported with batch-bootstrap CIs.

## Setup

```bash
pip install agentdq==0.1.0a1 pandas pyarrow scipy scikit-learn xgboost
```

## Olist

Needs the Olist Brazilian E-Commerce dataset (Kaggle, `olistbr/brazilian-ecommerce`) and the feature pipeline and frozen split of the companion study's repository, which builds `data/silver/order_features.parquet`. Then:

```bash
python run_experiments.py --repo <companion repo> --out results/olist --batches 1000 --rows 500
```

## OULAD

Download the Open University Learning Analytics Dataset (CC BY 4.0; UCI dataset 349 or the Open University site) as one zip.

```bash
python build_oulad.py --zip oulad.zip --out oulad_day28.parquet
python run_oulad.py --data oulad_day28.parquet --out results/oulad --batches 1000 --rows 500
```

Split by presentation: train 2013B and 2013J, probe set 2014B, test 2014J. Students already unregistered by day 28 are dropped.

## LLM agent (OULAD)

Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT`, plus either `AZURE_OPENAI_API_KEY` or an `az login` session. Start with the 200-row pilot, then run the full experiment (about 30,000 calls with the defaults; identical rows are cached).

```bash
python run_oulad_agent.py --data oulad_day28.parquet --out results/oulad_agent --pilot
python run_oulad_agent.py --data oulad_day28.parquet --out results/oulad_agent
```

## Main results (RQ1, Spearman ρ with realised damage)

| Data | Consumer | agentdq | Failed-check count | Summed severity |
|---|---|---|---|---|
| Olist | tree | 0.52 | 0.41 | 0.43 |
| Olist | text | 0.77 | 0.60 | 0.63 |
| OULAD | tree | 0.84 | 0.64 | 0.67 |
| OULAD | engagement | 0.71 | 0.50 | 0.52 |
| OULAD (150 × 200 rows) | agent | 0.34 | 0.30 | 0.29 |

The first four rows use 1,000 batches of 500 rows. The agent is the most robust consumer (mean damage 0.007 AUROC vs 0.025 for the tree); 62% of batches that fail at least one check cost it under 0.01.
