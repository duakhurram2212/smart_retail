# Smart Retail Demand Forecasting 

## Problem Statement
Predict the unit sales of each product family in each store for the **next 7 days**. Good forecasts help stores avoid stock-outs and overstocking (especially perishables). Phase 1 covers everything before modeling: cleaning, EDA, insights and feature preparation.

## Dataset
- **Source:** [Kaggle – Store Sales: Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- **Files used (6):** `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`
- **Size:** 3,000,888 rows (one row = date × store × product family), 1 Jan 2013 – 15 Aug 2017, 54 stores, 33 product families
- **Target:** `sales` (units sold; `log1p(sales)` recommended for modeling)
- **Main features:** `onpromotion`, store info (city, state, type, cluster), holiday/event flags, transactions, oil price
- **Final dataset:** 62 columns (47 engineered features). Full data dictionary: [`data/processed/README.md`](data/processed/README.md)

**Download links**
- Raw data: see the Kaggle link above (place the 6 CSVs in `data/raw/`).
- `features_engineered.parquet` (105 MB) is over GitHub's size limit, so it is not in the repo. Generate it by running `notebooks/feature_engineering.ipynb`.
- `cleaned_merged.parquet` (20 MB) is included in `data/processed/`.

## Repository Structure
```
smart_retail/
├── data/
│   ├── raw/                  # Kaggle CSVs (not in repo, download link above)
│   └── processed/            # cleaned_merged.parquet, features_engineered.parquet, data dictionary
├── notebooks/
│   ├── cleaning_merging.ipynb
│   ├── eda.ipynb
│   └── feature_engineering.ipynb
├── outputs/figures/          # saved EDA plots
├── requirements.txt
└── README.md
```

(Phase 1: Data Preparation)
Cleaning, EDA and feature preparation for forecasting daily supermarket sales, using the Kaggle *Store Sales – Time Series Forecasting* (Corporación Favorita, Ecuador) dataset.

## Cleaning Summary
| Issue | What we did | Why |
|---|---|---|
| 6 separate tables | Merged into one table (no duplicates, no NaNs) | One row has all context |
| Oil: only working days, 43 NaN | Full daily calendar, forward-fill | No future information (no leakage) |
| Transactions missing on closed days (7,330) | Set to 0 | Store closed = true zero |
| Transactions gaps (118) and glitches (2) | Median of previous 7 days per store + `tx_missing` flag | Robust to bad values, past data only |
| Multiple holiday entries per date | Converted to flags (national / regional / local / event); excluded Work Day and transferred holidays | Avoids duplicating sales rows |
| Memory size | Smaller dtypes + category columns (881 MB → 126 MB in memory, 20 MB parquet) | Faster loading |
| Skewed target (skew 5.85) | `log1p` shown in EDA (skew 0.21) | Better for modeling |
| Early-period missing records | Flagged with `reliable_period` (from 15 Jul 2015) | Pre-June-2015 zeros are missing data |

## Key Findings
1. **Recording change in 2015:** PRODUCE, HOME CARE and BEVERAGES (and several others) were mostly recorded as zero before June 2015, so train on `reliable_period == 1`.
2. **Promotion data** only exists from April 2014; earlier zeros mean "not recorded".
3. **Seasonality:** clear weekly pattern (Sunday highest, Thursday lowest) and a December peak every year; 24 Dec is up to ~2× normal; 1 Jan has almost no sales (closed).
4. **Holidays/events:** national holidays give ~20% higher sales; the April 2016 earthquake caused a ~4-day spike (1.5–1.7×).
5. **Concentration:** GROCERY I, BEVERAGES and PRODUCE make ~67% of sales, so errors there matter most.
6. **Distribution:** target is heavily right-skewed; ~31% of all rows are zero (~16% in the reliable period).
7. **Correlations:** transactions and store sales are strongly related; oil price shows no visible relationship with sales; payday effect is small.
8. **Promotions:** 31 of 32 families sell more on promo rows, but this is a naive comparison, not the causal effect.

Each plot has a "Finding" note in `notebooks/eda.ipynb`; images are saved in `outputs/figures/`.

## Scaling / Normalization
Not applied. We plan to use tree-based models (XGBoost/LightGBM), which do not need scaled features. The skewed target is handled with `log1p(sales)`. If a linear model is used in Phase 2, features will be scaled after the train/test split.

## How to Run
```bash
git clone <your-repo-url>
cd smart_retail
pip install -r requirements.txt
# download the 6 Kaggle CSVs into data/raw/
jupyter notebook
```
Run in this order (each notebook: *Kernel → Restart & Run All*):
1. `notebooks/cleaning_merging.ipynb` → `data/processed/cleaned_merged.parquet`
2. `notebooks/eda.ipynb` → figures in `outputs/figures/`
3. `notebooks/feature_engineering.ipynb` → `data/processed/features_engineered.parquet`

## Team
Dua Khurram     23L-0726
Ayesha Shahid   23L-0589
Ahmad 
