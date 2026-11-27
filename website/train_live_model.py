"""Train the XGBoost used by the website's live forecast and save it.

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
    FEATURES,
    TEST_START,
    TRAIN_START,
    add_features,
    load_history,
    model_paths,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, 'data', 'processed', 'cleaned_merged.parquet')


def main():
    print('loading history')
    history = load_history(PARQUET)
    families = sorted(history['family'].unique())
    types = sorted(history['store_type'].unique())
    family_codes = {name: i for i, name in enumerate(families)}
    type_codes = {name: i for i, name in enumerate(types)}

    print('building clues')
    featured = add_features(history, family_codes, type_codes)
    train = featured[(featured['date'] >= TRAIN_START) & (featured['date'] < TEST_START)].copy()
    train = train.dropna(subset=['sales_lag_28', 'roll_mean_28'])
    test = featured[featured['date'] >= TEST_START].copy()

    print('train rows', len(train), 'test rows', len(test))
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        tree_method='hist',
        random_state=42,
        n_jobs=4,
    )
    model.fit(train[FEATURES], np.log1p(train['sales']))

    raw = model.predict(test[FEATURES])
    guess = np.expm1(np.clip(raw, 0, None))
    actual = test['sales'].to_numpy()
    wape = 100 * np.abs(guess - actual).sum() / np.maximum(actual.sum(), 1)
    baseline = test['sales_same_dow_mean_4w'].to_numpy()
    base_wape = 100 * np.abs(baseline - actual).sum() / np.maximum(actual.sum(), 1)
    print('test WAPE %', round(float(wape), 2), 'repeat-4-weeks WAPE %', round(float(base_wape), 2))

    model_path, meta_path = model_paths(ROOT)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    meta = {
        'family_codes': family_codes,
        'type_codes': type_codes,
        'features': FEATURES,
        'test_start': str(TEST_START.date()),
        'train_start': str(TRAIN_START.date()),
        'test_wape': round(float(wape), 2),
    }
    with open(meta_path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    print('saved', model_path)


if __name__ == '__main__':
    main()
