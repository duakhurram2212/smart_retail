# Smart Retail Demand Forecasting

Cleaning, EDA and feature preparation for forecasting daily supermarket sales, using the Kaggle *Store Sales – Time Series Forecasting* (Corporación Favorita, Ecuador) dataset.

**Phase 1 (Data Preparation)** covers everything before modeling: data cleaning, exploratory data analysis (EDA), insights and feature preparation.

## Problem Statement
Predict the unit sales of each product family in each store for the **next 7 days**. Good forecasts help stores avoid stock-outs and overstocking (especially perishables).

## Dataset
- **Source:** [Kaggle – Store Sales: Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- **Files used (6):** `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`
- **Size:** 3,000,888 rows (one row = date × store × product family), 1 Jan 2013 – 15 Aug 2017, 54 stores, 33 product families
- **Target:** `sales` (units sold; `log1p(sales)` recommended for modeling). This is a **regression** problem, so there is no class imbalance in the usual sense, but the target is **zero-inflated** (about 31% of all rows have zero sales, about 16% in the reliable period).
- **Main features:** `onpromotion`, store info (city, state, type, cluster), holiday/event flags, transactions, oil price
- **Final dataset:** 62 columns (47 engineered features). Full data dictionary: [`data/processed/README.md`](data/processed/README.md)

**Download links**
- Raw data: see the Kaggle link above (place the 6 CSVs in `data/raw/`).
- `features_engineered.parquet` (105 MB) is over GitHub's size limit, so it is not in the repo.Generate it by running `notebooks/feature_engineering.ipynb`.
- `cleaned_merged.parquet` (20 MB) is included in `data/processed/`.

## Repository Structure
```
smart_retail/
├── data/
│   ├── raw/                  # Kaggle CSVs (not in repo, download link above)
│   └── processed/            # cleaned_merged.parquet, features_engineered.parquet, data dictionary
├── notebooks/
│   ├── cleaning_merging.ipynb      # cleaning + merging the 6 tables
│   ├── eda.ipynb                   # EDA, plots, findings and key insights
│   └── feature_engineering.ipynb   # lag / rolling / calendar / promo features
├── outputs/figures/          # saved EDA plots (16 images, see the table below)
├── requirements.txt
└── README.md
```

## Cleaning Summary
| Issue | What we did | Why |
|---|---|---|
| 6 separate tables | Merged into one table (no duplicates, no NaNs) | One row has all context |
| Duplicates | Checked `(date, store, family)` in train and `(date, store)` in transactions: none found | Each row must be unique |
| Oil: only working days, 43 NaN | Full daily calendar, forward-fill | No future information (no leakage) |
| Transactions missing on closed days (7,330) | Set to 0 | Store closed = true zero |
| Transactions gaps (118) and glitches (2) | Median of previous 7 days per store + `tx_missing` flag | Robust to bad values, past data only |
| Missing dates in train (25 Dec 2013–2016) | Not added | Stores were closed; lag features use real calendar dates, so nothing shifts |
| Multiple holiday entries per date | Converted to flags (national / regional / local / event); excluded Work Day and transferred holidays | Avoids duplicating sales rows |
| Outliers | Only the 2 transaction glitches were data errors and were fixed. High sales values were **kept** | They are real demand spikes (holidays, Christmas, promotions, earthquake); `log1p` handles the skew |
| Inconsistent formats | Parsed dates, renamed `type` to `store_type` and `cluster` to `store_cluster`, text columns to `category` (no negative values, no extra spaces) | Consistent names and types |
| Memory size | Smaller dtypes + category columns (881 MB → 126 MB in memory, 20 MB parquet) | Faster loading |
| Skewed target (skew 5.85) | `log1p` shown in EDA (skew 0.21) | Better for modeling |
| Early-period missing records | Flagged with `reliable_period` (from 15 Jul 2015) | Pre-June-2015 zeros are missing data. Lag/rolling features look back up to 35 days, so 15 Jul is a safe start after the 1 Jun 2015 recording change |

## Key Findings
1. **Recording change in 2015:** PRODUCE, HOME CARE and BEVERAGES (and several others) were mostly recorded as zero before June 2015, so train on `reliable_period == 1`.
2. **Promotion data** only exists from April 2014; earlier zeros mean "not recorded".
3. **Seasonality:** clear weekly pattern (Sunday highest, Thursday lowest) and a December peak every year; 24 Dec is up to ~2× normal; 1 Jan has almost no sales (closed).
4. **Holidays/events:** national holidays give ~20% higher sales; the April 2016 earthquake caused a ~4-day spike (1.5–1.7×).
5. **Concentration:** GROCERY I, BEVERAGES and PRODUCE make ~67% of sales, so errors there matter most.
6. **Distribution:** target is heavily right-skewed; ~31% of all rows are zero (~16% in the reliable period). Regression problem, so no class imbalance, but zero sales need attention.
7. **Correlations:** transactions and store daily sales are strongly related (r = 0.92). At row level, `onpromotion` has the highest correlation with log1p(sales) (0.43), then `transactions` (0.22). Oil price is very weak (r = 0.10 with total sales). Features are only weakly correlated with each other. Payday effect is small.
8. **Outliers:** inside each family, 3% to 11% of non-zero rows are above the boxplot whisker. They are real demand spikes, so they were kept.
9. **Stores:** store type and cluster matter. The median daily sales of a type A store (about 29.8k) is about 4× a type C store (about 7.2k), and the best cluster (5) sells about 6× more per row than the lowest (7).
10. **Promotions:** 31 of 32 families sell more on promo rows, but this is a naive comparison, not the causal effect.

The full list of insights is in the last cell of `notebooks/eda.ipynb`.

## Visualizations
All plots are shown in `notebooks/eda.ipynb` (each with a "Finding" note) and saved in `outputs/figures/`.

| Figure | What it tells us |
|---|---|
| `dailysales.png` | Total daily sales grow over time (about 0.35M to 0.85M per day) with a December peak. The noise before June 2015 is a recording problem, not real demand. |
| `produceanomaly.png` | PRODUCE is almost zero until mid-2015 and then steady, so earlier zeros are missing data. |
| `zeroshare.png` | Share of zero-sales rows by family and year. Several families are mostly zero in early years (missing records), while a few families truly sell rarely. |
| `seasonality.png` | Sunday is highest and Thursday lowest, December is the peak month, and sales keep growing year by year. |
| `topfamilies.png` | GROCERY I, BEVERAGES and PRODUCE make about 67% of sales. |
| `promoshare.png` | Promotions are 0% until early 2014, about 4% in 2014 Q2, and about 45% by late 2016. |
| `promostreaks.png` | Most promotions are short; a small group lasts 30 days or more. |
| `holidayearthquake.png` | National holidays give about +20% sales, event days about +4%, and the earthquake caused a ~4-day spike of 1.5–1.7×. |
| `christmas.png` | Sales rush on 23–24 Dec (24 Dec up to ~2× normal) and are almost zero on 1 Jan (closed). |
| `transactionoil.png` | Transactions and sales have a strong straight-line relationship (r = 0.92). Oil price and sales have almost none (r = 0.10). |
| `salesdistribution.png` | Sales are very right-skewed (5.85); after `log1p` the skew is 0.21. There is a spike at 0 (16.3% zero rows). |
| `correlation_heatmap.png` | `onpromotion` (0.43) and `transactions` (0.22) correlate most with log1p(sales); oil, holiday and event flags are near zero at row level. |
| `boxplot_family.png` | Families sell at very different levels. Points above the whiskers are real spikes, points at 0 are zero-sales rows. |
| `promovsnonpromo.png` | Almost all families sell more on promo rows, but this is a naive comparison, not the true effect. |
| `payday.png` | Month-end is about 4% above other days and day 15 is not higher, so payday is a minor factor. |
| `store_type_cluster.png` | Store type A and clusters 5 and 11 sell the most; type C and clusters 7 and 15 the least. |

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
The notebooks use relative paths (`../data/...`), so open them from inside the `notebooks/` folder.

Run in this order (each notebook: *Kernel → Restart & Run All*):
1. `notebooks/cleaning_merging.ipynb` → `data/processed/cleaned_merged.parquet` (needs the raw CSVs)
2. `notebooks/eda.ipynb` → figures in `outputs/figures/` (needs only `cleaned_merged.parquet`)
3. `notebooks/feature_engineering.ipynb` → `data/processed/features_engineered.parquet` (needs `cleaned_merged.parquet` and `holidays_events.csv`)

## Team
| Name | Roll No. |
|---|---|
| Dua Khurram | 23L-0726 |
| Ayesha Shahid | 23L-0589 |
| Ahmed Shaukat | 23L-0890 |
