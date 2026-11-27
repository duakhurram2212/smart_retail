"""Live 7-day forecast from the cleaned sales table.

The full 52-column feature file is not in the repo. These clues are built from
cleaned_merged.parquet at the moment of the forecast. Sales, transactions and
oil are taken from at least 7 days earlier, so a 7-day week never sees its own
sales. Promotion and holiday flags are kept, because a store knows those before
the week starts.
"""

import json
import os

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp('2017-07-19')
HISTORY_START = pd.Timestamp('2015-06-01')
TRAIN_START = pd.Timestamp('2015-07-15')
DATA_END = pd.Timestamp('2017-08-15')
FUTURE_START = pd.Timestamp('2017-08-16')
PLAN_COLUMNS = [
    'onpromotion',
    'is_holiday_national',
    'is_holiday_regional',
    'is_holiday_local',
    'is_event',
]

FEATURES = [
    'onpromotion',
    'is_holiday_national',
    'is_holiday_regional',
    'is_holiday_local',
    'is_event',
    'store_nbr',
    'family_code',
    'store_cluster',
    'store_type_code',
    'day_of_week',
    'month',
    'day',
    'is_weekend',
    'log_onpromotion',
    'sales_lag_7',
    'sales_lag_14',
    'sales_lag_21',
    'sales_lag_28',
    'sales_same_dow_mean_4w',
    'roll_mean_7',
    'roll_mean_28',
    'transactions_lag_7',
    'oil_lag_7',
]

HISTORY_COLUMNS = [
    'date', 'store_nbr', 'family', 'sales', 'onpromotion', 'store_type',
    'store_cluster', 'transactions', 'dcoilwtico', 'is_holiday_national',
    'is_holiday_regional', 'is_holiday_local', 'is_event',
]


def test_weeks():
    """Four scored weeks, then the week after the records stop."""
    future_end = FUTURE_START + pd.Timedelta(days=6)
    weeks = [(
        'Next week, sales unknown (' + FUTURE_START.strftime('%d %b') + ' to ' + future_end.strftime('%d %b %Y') + ')',
        FUTURE_START,
        future_end,
    )]
    for i in range(4):
        start = TEST_START + pd.Timedelta(days=7 * i)
        end = start + pd.Timedelta(days=6)
        label = 'Recorded week ' + str(i + 1) + ' (' + start.strftime('%d %b') + ' to ' + end.strftime('%d %b %Y') + ')'
        weeks.append((label, start, end))
    return weeks


def _store_family(history, store, family):
    part = history[(history['store_nbr'] == int(store)) & (history['family'] == str(family))].copy()
    part['date'] = pd.to_datetime(part['date']).dt.normalize()
    return part


def _ensure_days(part, week_start, week_end):
    """Add any missing day in the chosen week. Sales stay empty, so they cannot leak into the clues."""
    part = part.copy()
    part['date'] = pd.to_datetime(part['date']).dt.normalize()
    have = set(part['date'])
    template = part.sort_values('date').iloc[-1]
    extra = []
    for day in pd.date_range(week_start, week_end, freq='D'):
        day = pd.Timestamp(day).normalize()
        if day in have:
            continue
        row = template.copy()
        row['date'] = day
        row['sales'] = np.nan
        row['transactions'] = np.nan
        row['dcoilwtico'] = np.nan
        for col in PLAN_COLUMNS:
            row[col] = 0.0
        extra.append(row)
    if extra:
        part = pd.concat([part, pd.DataFrame(extra)], ignore_index=True)
    return part


def default_plan(history, store, family, week_start):
    """Promotion and holiday values currently on each day. Unknown days start at zero."""
    week_start = pd.Timestamp(week_start).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    part = _ensure_days(_store_family(history, store, family), week_start, week_end)
    days = pd.date_range(week_start, week_end, freq='D')
    rows = part[part['date'].isin(days)].sort_values('date')
    plan = rows[['date'] + PLAN_COLUMNS].copy()
    plan['onpromotion'] = plan['onpromotion'].fillna(0)
    for col in PLAN_COLUMNS[1:]:
        plan[col] = plan[col].fillna(0)
    return plan


def _apply_plan(part, plan):
    part = part.copy()
    part['date'] = pd.to_datetime(part['date']).dt.normalize()
    plan = plan.copy()
    plan['date'] = pd.to_datetime(plan['date']).dt.normalize()
    lookup = plan.set_index('date')
    for day, rec in lookup.iterrows():
        mask = part['date'] == pd.Timestamp(day).normalize()
        part.loc[mask, 'onpromotion'] = float(rec['onpromotion'])
        for col in PLAN_COLUMNS[1:]:
            part.loc[mask, col] = float(rec[col])
    return part


def load_history(parquet_path):
    df = pd.read_parquet(parquet_path, columns=HISTORY_COLUMNS)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= HISTORY_START].copy()
    df['family'] = df['family'].astype(str)
    df['store_nbr'] = df['store_nbr'].astype(int)
    df['store_type'] = df['store_type'].astype(str)
    df['store_cluster'] = df['store_cluster'].astype(int)
    for col in ['onpromotion', 'transactions', 'dcoilwtico', 'sales',
                'is_holiday_national', 'is_holiday_regional', 'is_holiday_local', 'is_event']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _calendar_lag(df, column, days, name):
    past = df[['store_nbr', 'family', 'date', column]].copy()
    past['date'] = past['date'] + pd.Timedelta(days=days)
    past = past.rename(columns={column: name})
    return past


def _rolling_mean_shifted(df, window):
    ordered = df.sort_values('date')
    rolled = (
        ordered.set_index('date')
        .groupby(['store_nbr', 'family'], sort=False)['sales']
        .rolling(str(window) + 'D')
        .mean()
        .reset_index()
    )
    rolled = rolled.rename(columns={'sales': 'roll_mean_' + str(window)})
    rolled['date'] = rolled['date'] + pd.Timedelta(days=7)
    return rolled


def add_features(df, family_codes, type_codes):
    df = df.sort_values(['store_nbr', 'family', 'date']).copy()
    out = df
    for days in (7, 14, 21, 28):
        lag = _calendar_lag(df, 'sales', days, 'sales_lag_' + str(days))
        out = out.merge(lag, on=['store_nbr', 'family', 'date'], how='left')
    tx = _calendar_lag(df, 'transactions', 7, 'transactions_lag_7')
    oil = _calendar_lag(df, 'dcoilwtico', 7, 'oil_lag_7')
    out = out.merge(tx, on=['store_nbr', 'family', 'date'], how='left')
    out = out.merge(oil, on=['store_nbr', 'family', 'date'], how='left')

    lag_cols = ['sales_lag_7', 'sales_lag_14', 'sales_lag_21', 'sales_lag_28']
    out['sales_same_dow_mean_4w'] = out[lag_cols].mean(axis=1)
    for window in (7, 28):
        rolled = _rolling_mean_shifted(df, window)
        out = out.merge(rolled, on=['store_nbr', 'family', 'date'], how='left')

    out['family_code'] = out['family'].map(family_codes).astype('float64')
    out['store_type_code'] = out['store_type'].map(type_codes).astype('float64')
    out['day_of_week'] = out['date'].dt.dayofweek.astype('float64')
    out['month'] = out['date'].dt.month.astype('float64')
    out['day'] = out['date'].dt.day.astype('float64')
    out['is_weekend'] = (out['date'].dt.dayofweek >= 5).astype('float64')
    out['log_onpromotion'] = np.log1p(out['onpromotion'].clip(lower=0))
    out['store_nbr'] = out['store_nbr'].astype('float64')
    out['store_cluster'] = out['store_cluster'].astype('float64')
    return out


def apply_codes(meta):
    return meta['family_codes'], meta['type_codes']


def model_paths(root):
    folder = os.path.join(root, 'models')
    return os.path.join(folder, 'xgb_live.json'), os.path.join(folder, 'live_meta.json')


def load_xgb(root):
    import xgboost as xgb

    model_path, meta_path = model_paths(root)
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(meta_path, encoding='utf-8') as handle:
        meta = json.load(handle)
    return model, meta


def predict_week(history, model, meta, store, family, week_start, plan=None):
    """Forecast 7 days for one store and family.

    plan can replace onpromotion and the holiday flags before the model runs.
    Days after 15 August 2017 have no sales in the data. Those rows are added
    with empty sales so the clues still come from 15 August and earlier.
    """
    week_start = pd.Timestamp(week_start).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    part = _ensure_days(_store_family(history, store, family), week_start, week_end)
    if plan is not None:
        part = _apply_plan(part, plan)
    featured = add_features(part, meta['family_codes'], meta['type_codes'])
    rows = featured[(featured['date'] >= week_start) & (featured['date'] <= week_end)].sort_values('date')
    if rows.empty:
        return rows
    raw = model.predict(rows[FEATURES])
    rows = rows.copy()
    rows['live_forecast'] = np.expm1(np.clip(raw, 0, None))
    rows['repeat_4_weeks'] = rows['sales_same_dow_mean_4w']
    rows['sales_known'] = rows['date'] <= DATA_END
    return rows
