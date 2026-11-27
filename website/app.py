import os
import pandas as pd
import streamlit as st

# paths (the app works from any folder)
app_folder = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(app_folder)
out = os.path.join(root, 'outputs')

st.set_page_config(page_title='Smart Retail', layout='wide')


@st.cache_data
def load_forecasts():
    # model predictions for the 4 test weeks
    preds = pd.read_csv(os.path.join(out, 'predictions', 'model_predictions.csv'), parse_dates=['date'])
    preds['family'] = preds['family'].astype(str)
    preds['store_nbr'] = preds['store_nbr'].astype(int)

    # real sales from the cleaned data
    sales = pd.read_parquet(os.path.join(root, 'data', 'processed', 'cleaned_merged.parquet'),
                            columns=['date', 'store_nbr', 'family', 'sales'])
    sales['date'] = pd.to_datetime(sales['date'])
    sales['family'] = sales['family'].astype(str)
    sales['store_nbr'] = sales['store_nbr'].astype(int)
    sales = sales[sales['date'] >= preds['date'].min() - pd.Timedelta(days=28)]

    table = preds.merge(sales, on=['date', 'store_nbr', 'family'], how='left')

    # baseline: average of the same weekday in the last 4 weeks
    lag_cols = []
    for k in [7, 14, 21, 28]:
        past = sales.copy()
        past['date'] = past['date'] + pd.Timedelta(days=k)
        past = past.rename(columns={'sales': 'lag_' + str(k)})
        table = table.merge(past, on=['date', 'store_nbr', 'family'], how='left')
        lag_cols.append('lag_' + str(k))
    table['pred_4week_avg'] = table[lag_cols].mean(axis=1)
    table = table.drop(columns=lag_cols)

    table['week'] = (table['date'] - table['date'].min()).dt.days // 7 + 1
    return table


@st.cache_data
def load_table(name):
    return pd.read_csv(os.path.join(out, 'tables', name))


def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption)


st.title('Smart Retail: Sales Forecasting and Promotion Analysis')
st.write('Data: Kaggle Store Sales (Corporacion Favorita, Ecuador). 54 stores, 33 product families.')

page = st.sidebar.radio('Page', ['Forecast', 'Model comparison', 'Promotions'])


# ---------------- page 1: forecast ----------------
if page == 'Forecast':
    st.header('Sales forecast for one store and product family')
    st.write('Pick a store, a product family and a test week. '
             'The app shows the real sales and the 7-day forecast of each model.')

    table = load_forecasts()

    store = st.selectbox('Store', sorted(table['store_nbr'].unique()))
    family = st.selectbox('Product family', sorted(table['family'].unique()))

    week_labels = {}
    for w in sorted(table['week'].unique()):
        days = table[table['week'] == w]['date']
        week_labels['Week ' + str(w) + ' (' + days.min().strftime('%d %b') + ' to ' + days.max().strftime('%d %b %Y') + ')'] = w
    week_name = st.selectbox('Test week', list(week_labels.keys()))
    week = week_labels[week_name]

    part = table[(table['store_nbr'] == store) & (table['family'] == family) & (table['week'] == week)]
    part = part.sort_values('date')

    show = pd.DataFrame()
    show['Date'] = part['date'].dt.strftime('%a %d %b')
    show['Real sales'] = part['sales'].values
    show['XGBoost'] = part['pred_xgboost'].values
    show['Regression'] = part['pred_regression'].values
    show['4-week average'] = part['pred_4week_avg'].values
    show = show.set_index('Date')

    st.line_chart(show)
    st.dataframe(show.round(1))

    st.subheader('Error for this selection (WAPE, lower is better)')
    total = show['Real sales'].sum()
    if total > 0:
        rows = []
        for model in ['XGBoost', 'Regression', '4-week average']:
            error = 100 * (show[model] - show['Real sales']).abs().sum() / total
            rows.append({'Model': model, 'WAPE %': round(error, 1)})
        st.dataframe(pd.DataFrame(rows).set_index('Model'))
    else:
        st.write('This store sold nothing of this family in this week, so the percentage error cannot be calculated.')


# ---------------- page 2: model comparison ----------------
if page == 'Model comparison':
    st.header('Model comparison')
    st.write('All models are tested on the last 4 weeks of the data (19 July to 15 August 2017), '
             'one week at a time. Models were trained only on data before 19 July.')

    summary = load_table('eval_summary.csv').set_index('model')
    st.subheader('Average over the 4 test weeks')
    st.dataframe(summary.round(3))
    st.write('XGBoost has the lowest error. Its WAPE is 12.6% against 15.6% for the 4-week average. '
             'The regression is about the same as the simple average.')

    by_block = load_table('eval_by_block.csv')
    st.subheader('RMSLE for each test week')
    st.dataframe(by_block.pivot(index='block', columns='model', values='RMSLE').round(3))

    show_image(os.path.join(out, 'figures_eval', 'fig1_model_comparison.png'), 'Model comparison')

    st.subheader('Error of the best model (XGBoost) by product family')
    by_family = load_table('eval_error_by_family.csv').set_index('family')
    st.dataframe(by_family.round(2))
    st.write('Very small families have large percentage errors, but they are almost 0% of total sales.')

    show_image(os.path.join(root, 'outputs', 'figures', 'shap_comparison.png'),
               'Feature importance (SHAP): regression vs XGBoost')


# ---------------- page 3: promotions ----------------
if page == 'Promotions':
    st.header('How much do promotions really help?')
    st.write('We compare sales before and during a promotion, and we also use stores with no promotion '
             'on the same days to remove normal changes in sales (weekdays, holidays, growth).')
    st.write('Overall result: the naive lift is 7.5%, but the real (adjusted) lift is about 4% '
             '(4.2% for promotion launches, 3.8% for weekly promotion surges). '
             'Longer promotions give a bigger lift.')

    launch = load_table('lift_by_family_launch.csv')
    weekly = load_table('lift_by_family_weekly.csv')

    families = sorted(set(launch['family']) | set(weekly['family']))
    family = st.selectbox('Product family', families)

    rows = []
    for name, data in [('Launches (Method A)', launch), ('Weekly surges (Method B)', weekly)]:
        r = data[data['family'] == family]
        if len(r) > 0:
            r = r.iloc[0]
            rows.append({'Method': name,
                         'Promotions used': int(r['n']),
                         'Naive lift %': round(r['naive_pct'], 1),
                         'Real lift %': round(r['adj_pct'], 1),
                         '95% CI low': round(r['adj_low'], 1),
                         '95% CI high': round(r['adj_high'], 1)})
    st.dataframe(pd.DataFrame(rows).set_index('Method'))

    if family in ['SCHOOL AND OFFICE SUPPLIES', 'LAWN AND GARDEN']:
        st.write('Note: this family has few promotions and may be affected by season, '
                 'so this number is not reliable.')

    show_image(os.path.join(out, 'figures_promo', 'fig1_lift_by_length.png'), 'Lift by promotion length')

    st.header('Does a promotion on one family reduce sales of another?')
    st.write('We tested 162 family pairs. No pair showed a clear drop in sales, '
             'so cannibalization was not detected between families.')

    pairs = load_table('cannibalization_all_pairs.csv')
    promoted = st.selectbox('Family on promotion', sorted(pairs['A'].unique()))
    part = pairs[pairs['A'] == promoted].sort_values('adj_pct')
    show = pd.DataFrame()
    show['Other family'] = part['B'].values
    show['Change in its sales %'] = part['adj_pct'].round(1).values
    show['95% CI low'] = part['adj_low'].round(1).values
    show['95% CI high'] = part['adj_high'].round(1).values
    show['Significant'] = (part['p_bh'] < 0.05).map({True: 'yes', False: 'no'}).values
    st.dataframe(show.set_index('Other family'))

    show_image(os.path.join(out, 'figures_promo', 'fig6_all_pairs_heatmap.png'), 'All pairs')
