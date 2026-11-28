"""Live 7-day forecast for one store and one product family.

The saved model is the Phase 2 XGBoost: the 52 features in models/xgb_features.json,
502 trees, subsample 1.0. Sales, transactions and oil used as clues are at least
7 days old. Promotion and holiday flags are kept, because a store knows those
before the week starts.
"""

import json
import os

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp('2017-07-19')
CAL_START = pd.Timestamp('2013-01-01')
DATA_END = pd.Timestamp('2017-08-15')
FUTURE_START = pd.Timestamp('2017-08-16')
PERISHABLE = {
    'DAIRY', 'PRODUCE', 'MEATS', 'POULTRY', 'SEAFOOD', 'EGGS',
    'BREAD/BAKERY', 'PREPARED FOODS', 'DELI',
}
TYPE_CODES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
PLAN_COLUMNS = [
    'onpromotion',
    'is_holiday_national',
    'is_holiday_regional',
    'is_holiday_local',
    'is_event',
]
FEATURES = [
    'store_nbr', 'onpromotion', 'store_cluster', 'is_holiday_national',
    'is_holiday_regional', 'is_holiday_local', 'is_event', 'is_holiday_any',
    'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'year',
    'is_weekend', 'is_payday', 'is_new_year', 'days_to_christmas', 'is_earthquake',
    'is_mothers_day', 'days_to_next_holiday', 'days_since_prev_holiday',
    'sales_lag_7d', 'sales_lag_14d', 'sales_lag_21d', 'sales_lag_28d',
    'sales_same_dow_mean_4w', 'rolling_mean_7d', 'rolling_std_7d',
    'rolling_mean_14d', 'rolling_std_14d', 'rolling_mean_28d', 'rolling_std_28d',
    'promo_flag', 'log_onpromotion', 'onpromotion_lag_7d', 'promo_streak',
    'promo_days_7d', 'promo_days_28d', 'promo_intensity_vs_28d',
    'store_promo_families', 'family_promo_stores', 'transactions_lag_7d',
    'transactions_roll_7d', 'dcoilwtico_lag_7d', 'dcoilwtico_roll_28d',
    'store_type_code', 'family_code', 'city_code', 'state_code', 'is_perishable',
    'days_since_open',
]
HISTORY_COLUMNS = [
    'date', 'store_nbr', 'family', 'sales', 'onpromotion', 'store_type',
    'store_cluster', 'city', 'state', 'transactions', 'dcoilwtico',
    'is_holiday_national', 'is_holiday_regional', 'is_holiday_local',
    'is_event', 'is_holiday_any',
]
_CONTEXT = {}


def test_weeks():
    """Four held-out weeks, then the week after the records stop."""
    future_end = FUTURE_START + pd.Timedelta(days=6)
    weeks = [(
        'Next week, sales unknown (' + FUTURE_START.strftime('%d %b') + ' to '
        + future_end.strftime('%d %b %Y') + ')',
        FUTURE_START,
        future_end,
    )]
    for i in range(4):
        start = TEST_START + pd.Timedelta(days=7 * i)
        end = start + pd.Timedelta(days=6)
        label = (
            'Held-out week ' + str(i + 1) + ' (' + start.strftime('%d %b')
            + ' to ' + end.strftime('%d %b %Y') + ')'
        )
        weeks.append((label, start, end))
    return weeks


def _store_family(history, store, family):
    part = history[
        (history['store_nbr'].astype(int) == int(store)) & (history['family'].astype(str) == str(family))
    ].copy()
    part['date'] = pd.to_datetime(part['date']).dt.normalize()
    return part


def _ensure_days(part, week_start, week_end):
    """Add any missing day in the chosen week. Sales stay empty so they cannot leak into the clues."""
    part = part.copy()
    part['date'] = pd.to_datetime(part['date']).dt.normalize()
    if part.empty:
        return part
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
        if 'is_holiday_any' in row.index:
            row['is_holiday_any'] = 0.0
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
        if 'is_holiday_any' in part.columns:
            part.loc[mask, 'is_holiday_any'] = max(
                float(rec['is_holiday_national']),
                float(rec['is_holiday_regional']),
                float(rec['is_holiday_local']),
            )
    return part


def load_history(parquet_path):
    df = pd.read_parquet(parquet_path, columns=HISTORY_COLUMNS)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['family'] = df['family'].astype(str)
    df['city'] = df['city'].astype(str)
    df['state'] = df['state'].astype(str)
    df['store_type'] = df['store_type'].astype(str)
    df['store_nbr'] = df['store_nbr'].astype(int)
    df['store_cluster'] = df['store_cluster'].astype(int)
    for col in [
        'onpromotion', 'transactions', 'dcoilwtico', 'sales',
        'is_holiday_national', 'is_holiday_regional', 'is_holiday_local',
        'is_event', 'is_holiday_any',
    ]:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    return df


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


def _context(history):
    key = id(history)
    hit = _CONTEXT.get(key)
    if hit is not None:
        return hit
    _CONTEXT.clear()
    frame = history[['store_nbr', 'family', 'date', 'onpromotion']].copy()
    frame['store_nbr'] = frame['store_nbr'].astype(int)
    frame['family'] = frame['family'].astype(str)
    frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
    frame['flag'] = (frame['onpromotion'] > 0).astype('int16')
    oil = (
        history.drop_duplicates('date')
        .assign(date=lambda frame: pd.to_datetime(frame['date']).dt.normalize())
        .set_index('date')['dcoilwtico']
        .sort_index()
    )
    tx = (
        history.assign(date=lambda frame: pd.to_datetime(frame['date']).dt.normalize())
        .drop_duplicates(['store_nbr', 'date'])
        .pivot(index='date', columns='store_nbr', values='transactions')
        .sort_index()
    )
    tx.columns = tx.columns.astype(int)
    tx.index = pd.to_datetime(tx.index).normalize()
    oil.index = pd.to_datetime(oil.index).normalize()

    def _multi(grouped, key_name):
        counts = grouped.sum()
        keys = counts.index.get_level_values(0)
        if key_name == 'store':
            keys = keys.astype(int)
        else:
            keys = keys.astype(str)
        dates = pd.to_datetime(counts.index.get_level_values(1)).normalize()
        counts.index = pd.MultiIndex.from_arrays([keys, dates])
        return counts

    hit = {
        'store': _multi(frame.groupby(['store_nbr', 'date'], sort=False)['flag'], 'store'),
        'family': _multi(frame.groupby(['family', 'date'], sort=False)['flag'], 'family'),
        'oil': oil,
        'tx': tx,
    }
    _CONTEXT[key] = hit
    return hit


def _lookup_count(series, key):
    try:
        return int(series.loc[key])
    except KeyError:
        return 0


def calendar_features(dates, holiday_dates):
    index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    day = pd.Series(index)
    frame = pd.DataFrame(index=index)
    frame['day_of_week'] = index.dayofweek
    frame['day_of_month'] = index.day
    frame['week_of_year'] = index.isocalendar().week.astype(int).to_numpy()
    frame['month'] = index.month
    frame['quarter'] = index.quarter
    frame['year'] = index.year
    frame['is_weekend'] = (index.dayofweek >= 5).astype(int)
    frame['is_payday'] = ((index.day == 15) | index.is_month_end).astype(int)
    frame['is_new_year'] = ((index.month == 1) & (index.day == 1)).astype(int)
    christmas = pd.to_datetime(day.dt.year.astype(str) + '-12-25')
    christmas_next = pd.to_datetime((day.dt.year + 1).astype(str) + '-12-25')
    frame['days_to_christmas'] = (christmas.where(day <= christmas, christmas_next) - day).dt.days.to_numpy()
    frame['is_earthquake'] = (
        (index >= pd.Timestamp('2016-04-16')) & (index <= pd.Timestamp('2016-05-16'))
    ).astype(int)
    frame['is_mothers_day'] = (
        (index.month == 5) & (index.dayofweek == 6) & (index.day >= 8) & (index.day <= 14)
    ).astype(int)
    holidays = pd.to_datetime(list(holiday_dates)).normalize().unique()
    holidays = np.sort(holidays.values.astype('datetime64[ns]'))
    values = index.values.astype('datetime64[ns]')
    if len(holidays) == 0:
        frame['days_to_next_holiday'] = 30
        frame['days_since_prev_holiday'] = 30
        return frame
    # A date after the last holiday, or before the first, is capped at 30.
    # Clamping the index onto the last holiday would turn that distance into 0.
    next_pos = np.searchsorted(holidays, values, side='left')
    prev_pos = np.searchsorted(holidays, values, side='right') - 1
    to_next = np.full(len(values), 30, dtype=int)
    since = np.full(len(values), 30, dtype=int)
    has_next = next_pos < len(holidays)
    has_prev = prev_pos >= 0
    to_next[has_next] = ((holidays[next_pos[has_next]] - values[has_next]) / np.timedelta64(1, 'D')).astype(int)
    since[has_prev] = ((values[has_prev] - holidays[prev_pos[has_prev]]) / np.timedelta64(1, 'D')).astype(int)
    frame['days_to_next_holiday'] = np.clip(to_next, 0, 30)
    frame['days_since_prev_holiday'] = np.clip(since, 0, 30)
    return frame


def history_features(sales, promo, index):
    sales = pd.Series(sales, copy=False).reindex(index).astype('float64')
    promo = pd.Series(promo, copy=False).reindex(index).fillna(0).astype('float64')
    out = pd.DataFrame(index=index)
    for lag in (7, 14, 21, 28):
        out['sales_lag_' + str(lag) + 'd'] = sales.shift(lag).to_numpy()
    lag_cols = ['sales_lag_' + str(lag) + 'd' for lag in (7, 14, 21, 28)]
    out['sales_same_dow_mean_4w'] = out[lag_cols].mean(axis=1).to_numpy()
    base = sales.shift(7)
    for win in (7, 14, 28):
        rolled = base.rolling(win, min_periods=int(win * 0.75))
        out['rolling_mean_' + str(win) + 'd'] = rolled.mean().to_numpy()
        out['rolling_std_' + str(win) + 'd'] = rolled.std().to_numpy()
    out['promo_flag'] = (promo.to_numpy() > 0).astype(np.int8)
    out['log_onpromotion'] = np.log1p(promo.clip(lower=0).to_numpy())
    out['onpromotion_lag_7d'] = promo.shift(7).to_numpy()
    flag = promo.to_numpy() > 0
    streak = np.zeros(len(index), dtype=np.int16)
    if len(streak):
        streak[0] = 1 if flag[0] else 0
        for t in range(1, len(streak)):
            streak[t] = streak[t - 1] + 1 if flag[t] else 0
    out['promo_streak'] = streak
    promo_days = (promo > 0).astype('float64')
    out['promo_days_7d'] = promo_days.rolling(7, min_periods=1).sum().to_numpy()
    out['promo_days_28d'] = promo_days.rolling(28, min_periods=1).sum().to_numpy()
    base28 = promo.shift(1).rolling(28, min_periods=7).mean()
    out['promo_intensity_vs_28d'] = (promo / (base28 + 1)).to_numpy()
    return out


def _series_on_calendar(values, index):
    """Reindex a date-indexed series onto a daily calendar. Gaps stay missing."""
    series = pd.Series(values, copy=False)
    series.index = pd.to_datetime(series.index).normalize()
    series = series[~series.index.duplicated(keep='last')].sort_index()
    return series.reindex(index)


def feature_rows(history, meta, store, family, week_start, plan=None):
    """52 model columns for the 7 days starting at week_start."""
    week_start = pd.Timestamp(week_start).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    store = int(store)
    family = str(family)
    part = _ensure_days(_store_family(history, store, family), week_start, week_end)
    if part.empty:
        return part
    if plan is not None:
        part = _apply_plan(part, plan)
    part['date'] = pd.to_datetime(part['date']).dt.normalize()
    part = part.sort_values('date').drop_duplicates('date', keep='last')
    index = pd.date_range(CAL_START, max(week_end, part['date'].max()), freq='D')
    sales = _series_on_calendar(part.set_index('date')['sales'], index)
    promo = _series_on_calendar(part.set_index('date')['onpromotion'], index).fillna(0)
    built = history_features(sales, promo, index)
    ctx = _context(history)
    tx_col = ctx['tx'][store] if store in ctx['tx'].columns else pd.Series(dtype='float64')
    tx_on_cal = _series_on_calendar(tx_col, index)
    tx_lag = tx_on_cal.shift(7)
    tx_roll = tx_on_cal.shift(7).rolling(7, min_periods=5).mean()
    oil_on_cal = _series_on_calendar(ctx['oil'], index).ffill()
    oil_lag = oil_on_cal.shift(7)
    oil_roll = oil_on_cal.shift(7).rolling(28, min_periods=14).mean()
    flags = part.set_index('date')[[
        'is_holiday_national', 'is_holiday_regional', 'is_holiday_local', 'is_event', 'is_holiday_any',
    ]]
    flags = _series_frame(flags, index).fillna(0)
    cal = calendar_features(index, meta['national_holidays'])
    info = part.iloc[0]
    open_date = pd.Timestamp(meta['open_dates'][str(store)])
    week_index = pd.date_range(week_start, week_end, freq='D')
    original = _store_family(history, store, family).drop_duplicates('date').set_index('date')['onpromotion']
    store_breadth = []
    family_breadth = []
    for day in week_index:
        new_flag = 1 if float(promo.loc[day]) > 0 else 0
        old_value = original.loc[day] if day in original.index else 0
        old_flag = 1 if float(old_value) > 0 else 0
        store_breadth.append(_lookup_count(ctx['store'], (store, day)) + new_flag - old_flag)
        family_breadth.append(_lookup_count(ctx['family'], (family, day)) + new_flag - old_flag)
    rows = pd.DataFrame(index=week_index)
    rows['store_nbr'] = store
    rows['onpromotion'] = promo.loc[week_index].to_numpy()
    rows['store_cluster'] = int(info['store_cluster'])
    for col in ['is_holiday_national', 'is_holiday_regional', 'is_holiday_local', 'is_event', 'is_holiday_any']:
        rows[col] = flags.loc[week_index, col].to_numpy()
    for col in [
        'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'year',
        'is_weekend', 'is_payday', 'is_new_year', 'days_to_christmas', 'is_earthquake',
        'is_mothers_day', 'days_to_next_holiday', 'days_since_prev_holiday',
    ]:
        rows[col] = cal.loc[week_index, col].to_numpy()
    for col in built.columns:
        rows[col] = built.loc[week_index, col].to_numpy()
    rows['store_promo_families'] = store_breadth
    rows['family_promo_stores'] = family_breadth
    rows['transactions_lag_7d'] = tx_lag.loc[week_index].to_numpy()
    rows['transactions_roll_7d'] = tx_roll.loc[week_index].to_numpy()
    rows['dcoilwtico_lag_7d'] = oil_lag.loc[week_index].to_numpy()
    rows['dcoilwtico_roll_28d'] = oil_roll.loc[week_index].to_numpy()
    rows['store_type_code'] = int(meta['type_codes'][str(info['store_type'])])
    rows['family_code'] = int(meta['family_codes'][family])
    rows['city_code'] = int(meta['city_codes'][str(info['city'])])
    rows['state_code'] = int(meta['state_codes'][str(info['state'])])
    rows['is_perishable'] = 1 if family in PERISHABLE else 0
    rows['days_since_open'] = np.clip((week_index - open_date).days, 0, None)
    rows['sales'] = sales.loc[week_index].to_numpy()
    rows['date'] = week_index
    return rows.reset_index(drop=True)


def _series_frame(frame, index):
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).normalize()
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    return frame.reindex(index)


def predict_week(history, model, meta, store, family, week_start, plan=None):
    """Forecast 7 days for one store and family."""
    rows = feature_rows(history, meta, store, family, week_start, plan)
    if rows.empty:
        return rows
    raw = model.predict(rows[FEATURES].astype('float32'))
    rows = rows.copy()
    rows['live_forecast'] = np.expm1(np.clip(raw, 0, None))
    rows['repeat_4_weeks'] = rows['sales_same_dow_mean_4w']
    rows['sales_known'] = rows['date'] <= DATA_END
    return rows
