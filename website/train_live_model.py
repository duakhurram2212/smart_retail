"""Train the website XGBoost so it matches the Phase 2 model.

Same specification as phase2/train_models.ipynb:
52 features, 502 trees, max_depth 10, learning_rate 0.03,
subsample 1.0, colsample_bytree 0.8, min_child_weight 1.
502 is the tree count early stopping chose in phase2/modeling.ipynb.
The final fit does not use a second early-stopping pass.

Run from the project root:

    py website/train_live_model.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_model import (
    CAL_START,
    FEATURES,
    PERISHABLE,
    TEST_START,
    TYPE_CODES,
    calendar_features,
    feature_rows,
    history_features,
    load_history,
    model_paths,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, 'data', 'processed', 'cleaned_merged.parquet')
SERIES_COLUMNS = [
    'sales_lag_7d', 'sales_lag_14d', 'sales_lag_21d', 'sales_lag_28d',
    'sales_same_dow_mean_4w', 'rolling_mean_7d', 'rolling_std_7d',
    'rolling_mean_14d', 'rolling_std_14d', 'rolling_mean_28d', 'rolling_std_28d',
    'promo_flag', 'log_onpromotion', 'onpromotion_lag_7d', 'promo_streak',
    'promo_days_7d', 'promo_days_28d', 'promo_intensity_vs_28d',
]


def _code_map(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = [str(value) for value in series.cat.categories]
    else:
        categories = sorted(set(series.astype(str)))
    return {name: i for i, name in enumerate(categories)}


def _open_dates(df):
    daily = df.groupby(['store_nbr', 'date'], sort=False)['sales'].sum().reset_index()
    opened = daily[daily['sales'] > 0].groupby('store_nbr')['date'].min()
    return {int(store): pd.Timestamp(day) for store, day in opened.items()}


def _add_series_features(df):
    """Sales and promotion history, one store-family at a time, same function the website uses."""
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
    columns = {name: np.full(len(df), np.nan, dtype='float64') for name in SERIES_COLUMNS}
    grouped = df.groupby(['store_nbr', 'family'], sort=False)
    total = grouped.ngroups
    for i, ((store, family), part) in enumerate(grouped):
        if i % 300 == 0:
            print('series', i, '/', total, flush=True)
        index = pd.date_range(CAL_START, part['date'].max(), freq='D')
        keyed = part.drop_duplicates('date').set_index('date')
        built = history_features(keyed['sales'], keyed['onpromotion'], index)
        taken = built.loc[pd.DatetimeIndex(part['date'])]
        positions = part.index.to_numpy()
        for name in SERIES_COLUMNS:
            columns[name][positions] = taken[name].to_numpy(dtype='float64')
    for name, values in columns.items():
        df[name] = values
    print('series', total, '/', total, flush=True)
    return df


def engineer(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['family'] = df['family'].astype(str)
    df['city'] = df['city'].astype(str)
    df['state'] = df['state'].astype(str)
    df['store_type'] = df['store_type'].astype(str)
    df['store_nbr'] = df['store_nbr'].astype(int)

    holidays = df.loc[df['is_holiday_national'] > 0, 'date'].drop_duplicates()
    # 25 Dec is a national holiday, but those rows were not stored because the stores were closed.
    christmas = [pd.Timestamp(f'{year}-12-25') for year in range(2013, 2017)]
    holidays = pd.to_datetime(list(holidays) + christmas).drop_duplicates().sort_values()
    calendar = calendar_features(pd.date_range(CAL_START, df['date'].max(), freq='D'), holidays)
    calendar = calendar.reset_index().rename(columns={'index': 'date'})
    before = len(df)
    df = df.merge(calendar, on='date', how='left')
    if len(df) != before:
        raise RuntimeError('calendar merge changed the row count')

    # Category order is the one stored in the parquet, which is what Phase 2 used.
    family_codes = _code_map(pd.read_parquet(PARQUET, columns=['family'])['family'])
    city_codes = _code_map(pd.read_parquet(PARQUET, columns=['city'])['city'])
    state_codes = _code_map(pd.read_parquet(PARQUET, columns=['state'])['state'])

    df['family_code'] = df['family'].map(family_codes).astype(int)
    df['city_code'] = df['city'].map(city_codes).astype(int)
    df['state_code'] = df['state'].map(state_codes).astype(int)
    df['store_type_code'] = df['store_type'].map(TYPE_CODES).astype(int)
    df['is_perishable'] = df['family'].isin(PERISHABLE).astype(int)

    opened = _open_dates(df)
    df['days_since_open'] = (df['date'] - df['store_nbr'].map(opened)).dt.days.clip(lower=0)
    df['is_store_open'] = (df['date'] >= df['store_nbr'].map(opened)).astype(int)
    df['reliable_period'] = (df['date'] >= pd.Timestamp('2015-07-15')).astype(int)

    print('building sales and promotion history', flush=True)
    df = _add_series_features(df)
    df['promo_flag'] = (df['onpromotion'] > 0).astype(int)
    df['store_promo_families'] = df.groupby(['store_nbr', 'date'], sort=False)['promo_flag'].transform('sum')
    df['family_promo_stores'] = df.groupby(['family', 'date'], sort=False)['promo_flag'].transform('sum')

    print('building transactions and oil', flush=True)
    full = pd.date_range(CAL_START, df['date'].max(), freq='D')
    tx = (
        df.drop_duplicates(['store_nbr', 'date'])
        .pivot(index='date', columns='store_nbr', values='transactions')
        .reindex(full)
    )
    tx_lag = tx.shift(7).stack().rename('transactions_lag_7d').reset_index()
    tx_roll = tx.shift(7).rolling(7, min_periods=5).mean().stack().rename('transactions_roll_7d').reset_index()
    tx_lag.columns = ['date', 'store_nbr', 'transactions_lag_7d']
    tx_roll.columns = ['date', 'store_nbr', 'transactions_roll_7d']
    before = len(df)
    df = df.merge(tx_lag, on=['date', 'store_nbr'], how='left')
    df = df.merge(tx_roll, on=['date', 'store_nbr'], how='left')
    if len(df) != before:
        raise RuntimeError('transaction merge changed the row count')
    oil = df.drop_duplicates('date').set_index('date')['dcoilwtico'].reindex(full).ffill()
    df['dcoilwtico_lag_7d'] = df['date'].map(oil.shift(7))
    df['dcoilwtico_roll_28d'] = df['date'].map(oil.shift(7).rolling(28, min_periods=14).mean())

    meta = {
        'family_codes': family_codes,
        'type_codes': TYPE_CODES,
        'city_codes': city_codes,
        'state_codes': state_codes,
        'open_dates': {str(store): str(day.date()) for store, day in opened.items()},
        'national_holidays': [str(pd.Timestamp(day).date()) for day in holidays],
        'features': FEATURES,
        'test_start': str(TEST_START.date()),
        'n_estimators': 502,
        'max_depth': 10,
        'learning_rate': 0.03,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
    }
    return df, meta


def _check_against_live(featured, meta):
    """The page and this script must build the same 52 columns."""
    history = load_history(PARQUET)
    checks = [(1, 'GROCERY I', TEST_START), (10, 'BEAUTY', pd.Timestamp('2017-03-01'))]
    worst = 0.0
    worst_name = ''
    for store, family, start in checks:
        got = feature_rows(history, meta, store, family, start).sort_values('date')
        end = start + pd.Timedelta(days=6)
        truth = featured[
            (featured['store_nbr'] == int(store))
            & (featured['family'] == family)
            & (featured['date'] >= start)
            & (featured['date'] <= end)
        ].sort_values('date')
        both = got.merge(truth, on='date', suffixes=('_live', '_train'))
        if len(both) < 6:
            raise RuntimeError('too few overlapping days for store ' + str(store) + ' ' + family)
        for name in FEATURES:
            left = both[name + '_live'].to_numpy(dtype='float64')
            right = both[name + '_train'].to_numpy(dtype='float64')
            both_missing = np.isnan(left) & np.isnan(right)
            gap = np.abs(left - right)
            gap[both_missing] = 0
            if np.isnan(gap).any():
                raise RuntimeError(name + ' has a NaN on only one side for store ' + str(store))
            diff = float(np.nanmax(gap)) if len(gap) else 0.0
            if diff > worst:
                worst = diff
                worst_name = name + ' store ' + str(store) + ' ' + family
    print('largest feature gap', round(worst, 6), worst_name, flush=True)
    if worst > 1e-3:
        raise RuntimeError('live features do not match training features')


def _weekly_wape(dates, actual, guess, start):
    order = np.argsort(dates)
    dates = dates[order]
    actual = actual[order]
    guess = guess[order]
    week = ((dates - np.datetime64(start)) / np.timedelta64(1, 'D')).astype(int) // 7
    scores = []
    for item in sorted(set(week)):
        mask = week == item
        total = actual[mask].sum()
        if total <= 0:
            continue
        scores.append(100 * np.abs(guess[mask] - actual[mask]).sum() / total)
    return float(np.mean(scores))


def main():
    print('loading', flush=True)
    raw = pd.read_parquet(PARQUET)
    featured, meta = engineer(raw)
    print('checking the live feature builder', flush=True)
    _check_against_live(featured, meta)

    usable = featured[(featured['reliable_period'] == 1) & (featured['is_store_open'] == 1)].copy()
    train = usable[usable['date'] < TEST_START]
    valid = usable[usable['date'] >= TEST_START]
    print('train rows', len(train), 'test rows', len(valid), flush=True)
    model = xgb.XGBRegressor(
        n_estimators=502,
        max_depth=10,
        learning_rate=0.03,
        subsample=1.0,
        colsample_bytree=0.8,
        min_child_weight=1,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES].astype('float32'), np.log1p(train['sales'].to_numpy()))
    raw_pred = model.predict(valid[FEATURES].astype('float32'))
    guess = np.expm1(np.clip(raw_pred, 0, None))
    actual = valid['sales'].to_numpy(dtype='float64')
    dates = valid['date'].to_numpy()
    wape = _weekly_wape(dates, actual, guess, TEST_START)
    log_actual = np.log1p(actual)
    log_guess = np.clip(raw_pred, 0, None)
    rmsle = float(np.sqrt(np.mean((log_guess - log_actual) ** 2)))
    print('mean weekly WAPE %', round(wape, 2), 'RMSLE', round(rmsle, 4), flush=True)
    meta['test_wape'] = round(wape, 2)
    meta['test_rmsle'] = round(rmsle, 4)

    model_path, meta_path = model_paths(ROOT)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    with open(meta_path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    with open(os.path.join(ROOT, 'models', 'xgb_features.json'), 'w', encoding='utf-8') as handle:
        json.dump(FEATURES, handle)
    print('saved', model_path, flush=True)


if __name__ == '__main__':
    main()
