# Smart Retail Demand Forecasting

Cleaning, EDA and feature preparation for forecasting daily supermarket sales, using the Kaggle *Store Sales – Time Series Forecasting* (Corporación Favorita, Ecuador) dataset.

**Phase 1 (Data Preparation)** covers everything before modeling: data cleaning, exploratory data analysis (EDA), insights and feature preparation.

**Phase 2 (Modeling, Evaluation and Website)** covers the forecasting models, their comparison on a held-out test period, the promotion impact analysis, and a web app that shows the results.

Live website: [https://smartretaill.streamlit.app/](https://smartretaill.streamlit.app/)

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
- `features_engineered.parquet` (105 MB) is over GitHub's size limit, so it is not in the repo.Generate it by running `phase1/feature_engineering.ipynb`.
- `cleaned_merged.parquet` (20 MB) is included in `data/processed/`.

## Repository Structure
```
smart_retail/
├── data/
│   ├── raw/                  # Kaggle CSVs (not in repo, download link above)
│   └── processed/            # cleaned_merged.parquet, features_engineered.parquet, data dictionary
├── phase1/                   # Phase 1: data preparation
│   ├── cleaning_merging.ipynb      # cleaning + merging the 6 tables
│   ├── eda.ipynb                   # EDA, plots, findings and key insights
│   └── feature_engineering.ipynb   # lag / rolling / calendar / promo features
├── phase2/                   # Phase 2: modeling and evaluation
│   ├── modeling.ipynb              # regression (gradient descent) and XGBoost, SHAP
│   ├── promo_lift_cannibalization.ipynb  # promotion lift and cannibalization
│   └── evaluation_metrics.ipynb    # all models and baselines on the same test weeks
├── models/                   # trained model used by the website
│   ├── xgb_live.json               # trained XGBoost (live forecast)
│   └── live_meta.json              # feature list, codes and test score of that model
├── website/                  # web app (see "Website" below)
│   ├── app.py                      # Streamlit app (4 pages)
│   ├── live_model.py               # builds the model inputs and makes the forecast
│   ├── train_live_model.py         # trains the model and saves it to models/
│   ├── requirements.txt            # libraries used by the deployed app
│   └── *.jpg                       # page photos
├── .streamlit/
│   └── config.toml           # app colours
├── outputs/
│   ├── figures/              # EDA plots (Phase 1) and SHAP plots (Phase 2)
│   ├── figures_promo/        # promotion lift and cannibalization plots
│   ├── figures_eval/         # model comparison plots
│   ├── tables/               # result tables (CSV)
│   ├── predictions/          # model_predictions.csv (test period)
│   └── metrics_table.csv     # scores of the regression models and XGBoost
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

The full list of insights is in the last cell of `phase1/eda.ipynb`.

## Visualizations
All plots are shown in `phase1/eda.ipynb` (each with a "Finding" note) and saved in `outputs/figures/`.

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
git clone https://github.com/duakhurram2212/smart_retail.git
cd smart_retail
pip install -r requirements.txt
# download the 6 Kaggle CSVs into data/raw/
jupyter notebook
```
The notebooks use relative paths (`../data/...`, `../outputs/...`), so open each notebook from inside its own folder (`phase1/` or `phase2/`).

Run in this order (each notebook: *Kernel > Restart & Run All*):
1. `phase1/cleaning_merging.ipynb` -> `data/processed/cleaned_merged.parquet` (needs the raw CSVs)
2. `phase1/eda.ipynb` -> figures in `outputs/figures/` (needs only `cleaned_merged.parquet`)
3. `phase1/feature_engineering.ipynb` -> `data/processed/features_engineered.parquet` (needs `cleaned_merged.parquet` and `holidays_events.csv`)
4. `phase2/modeling.ipynb` -> `outputs/predictions/model_predictions.csv`, `outputs/metrics_table.csv`, SHAP plots (needs `features_engineered.parquet`)
5. `phase2/promo_lift_cannibalization.ipynb` -> `outputs/figures_promo/` and `outputs/tables/` (needs `features_engineered.parquet` and `data/raw/train.csv`)
6. `phase2/evaluation_metrics.ipynb` -> `outputs/figures_eval/` and `outputs/tables/eval_*.csv` (needs `model_predictions.csv` and `features_engineered.parquet`)

## Phase 2: Forecasting Models (Ayesha)
Notebook: `phase2/modeling.ipynb`

- **Target:** `log1p(sales)`, predictions are converted back to normal sales before they are saved.
- **Split:** models are trained on data before 19 July 2017 and tested on the last 4 weeks (19 July to 15 August 2017).
- **Models:**
  - Linear regression trained with gradient descent, written from scratch (no regularization, L2, L1). Checked against scikit-learn.
  - XGBoost, tuned with a randomized search and early stopping.
- **Feature importance:** SHAP plots for XGBoost are in `outputs/figures/` (`shap_comparison.png`, `shap_xgb_summary.png`).

| Model | RMSLE | RMSE | MAPE % | R2 |
|---|---|---|---|---|
| GD (no regularization) | 0.4513 | 289.19 | 35.02 | 0.9474 |
| GD + L2 | 0.4512 | 288.82 | 35.03 | 0.9475 |
| GD + L1 | 0.4510 | 288.64 | 35.03 | 0.9476 |
| XGBoost | 0.3971 | 214.91 | 31.91 | 0.9707 |

The scores above come from `outputs/metrics_table.csv`. The full comparison with the baselines is in the evaluation section below.

## Phase 2: Promotion Impact and Model Evaluation (Ahmed)

This part answers two questions with the cleaned data from Phase 1, and then scores all models on the same test weeks.

1. How much extra sales does a promotion really bring? (promotion lift)
2. When one product family is on promotion, do sales of another family go down? (cannibalization)

### Promotion lift and cannibalization
Notebook: `phase2/promo_lift_cannibalization.ipynb`

**Method.** Comparing sales before and during a promotion is not fair, because sales also change due to weekdays, holidays, December and growth. So for every promotion we also look at other stores with no promotion on the same days. Their change is what we would expect anyway. The real lift is the difference. Confidence intervals use a bootstrap that resamples whole weeks.
- Method A (launches): the first day of a promotion after 7 days with no promotion, compared with the 7 days before.
- Method B (weekly surge): a week compared with the week before, when the number of promoted items jumps. This works for big families that are on promotion almost every day.

**Main results**
| Result | Value |
|---|---|
| Launches used (Method A) | 5,071 |
| Naive lift (before vs during only) | 7.5% |
| Adjusted lift, with comparison stores | 4.2% (95% CI 1.1% to 8.6%) |
| Adjusted lift, weekly surges (Method B) | 3.8% (95% CI 1.6% to 6.2%) |
| 1 day promotions | about 3.5% |
| 7+ day promotions | about 16% |
| Cannibalization | not detected |

- The naive comparison overstates the lift. About half of it is normal change in sales, not the promotion.
- Longer promotions show a bigger lift than short ones.
- The size of the lift depends on how a launch is defined. With a 14 day "no promotion before" window the lift is 8.2%, and with a 3 day window it is 21.0%. It is positive every time. We use 7 days as the main setting.
- Cannibalization: we tested 8 chosen pairs (substitutes and complements) and then all 162 family pairs, with a Benjamini-Hochberg correction. 5 pairs are significant and all 5 are positive. No pair shows a clear drop in sales.

**Limits.** A product here is a whole product family, so we can only check cannibalization between families, not brands. `onpromotion` is a count of items, so we cannot measure the size of the discount. Stores choose when to promote, so some of the effect may be timing.

Figures are in `outputs/figures_promo/` and tables are in `outputs/tables/`.

### Evaluation of the forecasting models
Notebook: `phase2/evaluation_metrics.ipynb`

All models are scored on the same test period: the last 4 weeks of the data (19 July to 15 August 2017), one week at a time, because the forecast horizon is 7 days. Models were trained only on data before 19 July. Two simple baselines are added, so we can see if the models beat a simple guess:
- same day last week
- average of the same weekday over the last 4 weeks

Metrics: RMSLE, RMSE, MAPE (only rows with sales above 0), WAPE and R2. MAPE breaks on zero sales rows, so WAPE (total error / total sales) is the main percentage metric.

| Model | RMSLE | RMSE | WAPE % | R2 |
|---|---|---|---|---|
| XGBoost | 0.397 | 214.9 | 12.6 | 0.971 |
| Gradient Descent Regression (L1) | 0.446 | 261.1 | 16.1 | 0.957 |
| Baseline: 4-week average | 0.467 | 254.7 | 15.6 | 0.958 |
| Baseline: same day last week | 0.546 | 309.8 | 18.5 | 0.939 |

- XGBoost has the lowest error. Its WAPE is 12.6% against 15.6% for the best baseline, and it has the lowest RMSLE in all 4 test weeks. The live forecast uses a simplified version of the model (13.5% error); the scores in the evaluation come from the full model (12.6%).
- The regression is about the same as the simple 4-week average.
- Best error on the big families: PRODUCE 8.3%, DAIRY 8.7%, GROCERY I 10.1%. Weakest big family: CLEANING at 21.6%. Very small families (for example BOOKS) have huge percentage errors, but they are almost 0% of sales.

Figures are in `outputs/figures_eval/` and tables are in `outputs/tables/` (`eval_*.csv`). The predictions used are in `outputs/predictions/model_predictions.csv`.

## Website
Code is in the `website/` folder. The shop has four pages: Shop floor, Stock the shelf, Scoreboard, and Offers. Stock the shelf scores the saved model `models/xgb_live.json` for a store number (1–54), an aisle, and a date. Allowed dates are 1 August 2015 through 16 August 2017. The forecast is the next 7 days. The app does not retrain. Scoreboard and Offers show the saved study results. The live forecast uses a simplified version of the model (13.5% error); the scores in the evaluation come from the full model (12.6%).

### Run it on your computer
From the project folder, after `pip install -r requirements.txt`:

```bash
python -m streamlit run website/app.py
```

Then open http://localhost:8501.

The files it needs are already in the repo: `data/processed/cleaned_merged.parquet`, `models/xgb_live.json`, `models/live_meta.json`, and the CSV files in `outputs/tables/`.

### Live site
[https://smartretaill.streamlit.app/](https://smartretaill.streamlit.app/)

It is deployed on Streamlit Community Cloud from this repo (`main`, main file `website/app.py`). The cloud app installs `website/requirements.txt` and runs on Python 3.12.

## Final Results and Conclusion
- **Forecasting:** XGBoost is the best model. On the 4 test weeks its WAPE is 12.6%, against 15.6% for the best simple baseline (4-week same-weekday average) and 16.1% for the best regression. Its R2 is 0.971. It is most accurate on the big families (PRODUCE 8.3%, DAIRY 8.7%, GROCERY I 10.1%) and weakest on CLEANING (21.6%).
- **Promotions:** a promotion raises sales by about 4% on average (95% CI 1.1% to 8.6%) once normal changes in sales are removed. Longer promotions give a bigger lift. We did not find clear cannibalization between product families.
- **In practice:** a store can use the forecast to plan stock for the next 7 days, and can expect a small but positive gain from promotions.
- **Limits:** the promotion result depends on how a launch is defined and is not a controlled experiment. Forecast errors are large for very small families. The website scores a saved XGBoost model for one store and aisle. It does not retrain.

## Team
| Name | Roll No. |
|---|---|
| Dua Khurram | 23L-0726 |
| Ayesha Shahid | 23L-0589 |
| Ahmed Shaukat | 23L-0890 |
