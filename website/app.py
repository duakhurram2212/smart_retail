import base64
import importlib
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_model
importlib.reload(live_model)
from live_model import TEST_START, default_plan, test_weeks
from live_model import load_history as load_sales_history
from live_model import load_xgb, predict_week

# paths (the app works from any folder)
app_folder = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(app_folder)
out = os.path.join(root, 'outputs')

st.set_page_config(page_title='Favorita Market', page_icon='🛒', layout='wide', initial_sidebar_state='expanded')

AISLE_ICONS = {
    'AUTOMOTIVE': '🚗', 'BABY CARE': '🍼', 'BEAUTY': '💄', 'BEVERAGES': '🥤',
    'BOOKS': '📚', 'BREAD/BAKERY': '🍞', 'CELEBRATION': '🎉', 'CLEANING': '🧹',
    'DAIRY': '🥛', 'DELI': '🥪', 'EGGS': '🥚', 'FROZEN FOODS': '🧊',
    'GROCERY I': '🛒', 'GROCERY II': '🧺', 'HARDWARE': '🔧',
    'HOME AND KITCHEN I': '🍳', 'HOME AND KITCHEN II': '🍽️', 'HOME APPLIANCES': '📺',
    'HOME CARE': '🧴', 'LADIESWEAR': '👗', 'LAWN AND GARDEN': '🌱', 'LINGERIE': '👚',
    'LIQUOR,WINE,BEER': '🍷', 'MAGAZINES': '📰', 'MEATS': '🥩', 'PERSONAL CARE': '🧼',
    'PET SUPPLIES': '🐾', 'PLAYERS AND ELECTRONICS': '🎧', 'POULTRY': '🍗',
    'PREPARED FOODS': '🍱', 'PRODUCE': '🥬', 'SCHOOL AND OFFICE SUPPLIES': '✏️',
    'SEAFOOD': '🐟',
}

MODEL_NAMES = {
    'xgboost': 'XGBoost',
    'regression': 'Straight-line regression',
    '4week_avg': 'Repeat the last 4 weeks',
    'last_week': 'Copy last week',
}


@st.cache_data
def load_table(name):
    return pd.read_csv(os.path.join(out, 'tables', name))


@st.cache_resource
def load_live():
    return load_xgb(root)


@st.cache_data
def sales_history():
    return load_sales_history(os.path.join(root, 'data', 'processed', 'cleaned_merged.parquet'))


def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption)


def friendly_models(frame):
    renamed = frame.rename(index=MODEL_NAMES)
    return renamed.rename(columns=MODEL_NAMES)


def icon_for(name):
    return AISLE_ICONS.get(str(name), '🏷️')


@st.cache_data
def photo_src(filename):
    path = os.path.join(app_folder, filename)
    with open(path, 'rb') as handle:
        return 'data:image/jpeg;base64,' + base64.b64encode(handle.read()).decode()


def page_heading(title, tag, photo=None):
    text = '<div class="shop-brand">' + title + '</div><div class="shop-tag">' + tag + '</div>'
    if photo:
        st.markdown(
            '<div class="shop-banner has-photo">'
            '<img class="top-pic" src="' + photo_src(photo) + '" alt="' + title + '">'
            '<div class="shop-copy">' + text + '</div></div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        '<div class="shop-banner"><div class="awning"></div>' + text + '</div>',
        unsafe_allow_html=True,
    )


def shelf(names, selected=None):
    cards = []
    for name in names:
        picked = ' picked' if selected == name else ''
        cards.append(
            '<div class="aisle' + picked + '"><div class="emoji">' + icon_for(name) +
            '</div><div class="name">' + name + '</div></div>'
        )
    st.markdown('<div class="shelf">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; max-width: 1180px; }
    .shop-banner {
        background: linear-gradient(180deg, #0E7A45 0%, #14915A 100%);
        color: #FFF8EE;
        border-radius: 22px;
        padding: 22px 26px 20px;
        margin-bottom: 16px;
        box-shadow: 0 10px 0 #C96B2C;
    }
    .awning {
        height: 14px;
        margin: -22px -26px 16px;
        border-radius: 22px 22px 0 0;
        background: repeating-linear-gradient(90deg, #F4C7A1 0 28px, #FFF8EE 28px 56px);
    }
    .shop-banner.has-photo {
        position: relative;
        padding: 0;
        background: none;
        overflow: hidden;
    }
    .top-pic {
        width: 100%;
        height: 280px;
        object-fit: cover;
        object-position: center 40%;
        display: block;
    }
    .shop-copy {
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        padding: 28px 26px 18px;
        background: linear-gradient(transparent, rgba(8, 48, 28, 0.88));
    }
    .shop-brand { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; }
    .shop-tag { opacity: 0.92; margin-top: 4px; }
    .shelf {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(132px, 1fr));
        gap: 12px;
        margin: 8px 0 18px;
    }
    .aisle {
        background: #fff;
        border: 1px solid #EADDCB;
        border-radius: 16px;
        padding: 12px 8px 10px;
        text-align: center;
        box-shadow: 0 5px 0 #E7D7C3;
    }
    .aisle.picked { border-color: #0E7A45; box-shadow: 0 5px 0 #0E7A45; }
    .aisle .emoji { font-size: 1.7rem; }
    .aisle .name { font-size: 0.72rem; font-weight: 700; margin-top: 4px; letter-spacing: 0.02em; }
    .ticket {
        background: #fff;
        border: 1px dashed #C96B2C;
        border-radius: 14px;
        padding: 10px 6px;
        text-align: center;
    }
    .ticket .d { font-size: 0.75rem; color: #7A5B3A; }
    .ticket .u { font-size: 1.35rem; font-weight: 800; color: #0E7A45; }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #EADDCB;
        border-radius: 16px;
        padding: 12px 14px;
        box-shadow: 0 5px 0 #E7D7C3;
    }
    section[data-testid="stSidebar"] {
        background: #FFF8EE;
        border-right: 1px solid #EADDCB;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }
    .side-card {
        background: #0E7A45;
        color: #FFF8EE;
        border-radius: 18px;
        padding: 16px 14px 14px;
        margin-bottom: 14px;
        box-shadow: 0 6px 0 #C96B2C;
    }
    .side-mark { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.03em; }
    .side-sub { opacity: 0.9; font-size: 0.82rem; margin-top: 2px; }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 16px;
        padding: 0.7rem 0.9rem;
        font-weight: 700;
        border: 1px solid #EADDCB;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: #fff;
        color: #1C1917;
        box-shadow: 0 4px 0 #E7D7C3;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #0E7A45;
        color: #FFF8EE;
        border-color: #0E7A45;
        box-shadow: 0 4px 0 #0A5A34;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="side-card"><div class="side-mark">Favorita</div>'
    '<div class="side-sub">54 stores · 33 aisles</div></div>',
    unsafe_allow_html=True,
)
PAGES = ['🛒  Shop floor', '📦  Stock the shelf', '🆚  Scoreboard', '🏷️  Offers']
if 'page' not in st.session_state:
    st.session_state.page = PAGES[0]
for label in PAGES:
    kind = 'primary' if st.session_state.page == label else 'secondary'
    if st.sidebar.button(label, key=f'nav-{label}', type=kind, width='stretch'):
        st.session_state.page = label
        st.rerun()
page = st.session_state.page


# ---------------- start here ----------------
if page.endswith('Shop floor'):
    page_heading(
        'Favorita Market',
        '54 stores in Ecuador · 33 aisles · open in the data until 15 Aug 2017',
        'market_hero.jpg',
    )
    summary = load_table('eval_summary.csv').set_index('model')
    sens = load_table('lift_sensitivity.csv')
    pairs = load_table('cannibalization_all_pairs.csv')
    main = sens[sens['before days'] == 7].iloc[0]
    clear = pairs[pairs['p_bh'] < 0.05]
    drops = int((clear['adj_pct'] < 0).sum())

    xgb_wape = summary.loc['xgboost', 'WAPE %']
    simple_wape = summary.loc['4week_avg', 'WAPE %']

    c1, c2, c3 = st.columns(3)
    c1.metric('Shelf miss', f'{xgb_wape:.1f}%', help='Share of units the best model got wrong.')
    c2.metric('Offer bump', f'{main["adjusted %"]:.1f}%', help='Extra sales after removing normal ups and downs.')
    c3.metric('Aisles that lost sales', str(drops), help='Pairs where another aisle clearly sold less.')

    st.caption(
        f'For every 100 units sold, XGBoost was off by about {xgb_wape:.0f}. '
        f'A simple “same weekday, last 4 weeks” guess was off by about {simple_wape:.0f}. '
        f'The fair promotion lift is {main["adjusted %"]:.1f}%, and no family pair showed a clear drop in sales. '
        'Stock the shelf uses this same XGBoost.'
    )

    st.subheader('The 33 aisles')
    shelf(list(AISLE_ICONS.keys()))
    st.caption('Each card is a whole aisle, not one brand. GROCERY I, drinks and produce are most of the basket.')

    st.header('Why the shop needs a number')
    st.write(
        'The store does not need a chart of the past. It needs a number for next week: '
        'how many units of groceries, drinks, produce, and the other families to put on the shelf in each store. '
        'The second question is about discounts. A promotion looks good if you only compare the promo days with the days before. '
        'Those days also include weekends, holidays, and the fact that the chain was growing. '
        'We wanted the extra sales that are actually from the promotion.'
    )

    st.header('How we know')
    left, right = st.columns(2)
    with left:
        st.subheader('The forecast test')
        st.write(
            'We hid the last 4 weeks (19 July to 15 August 2017) and trained only on earlier days. '
            'Then we compared each guess with what the stores really sold. '
            'A model only counts as useful if it beats two boring guesses: copy last week, '
            'and repeat the average of that same weekday over the last 4 weeks.'
        )
    with right:
        st.subheader('The promotion test')
        st.write(
            'For every promotion we also looked at other stores that were not promoting on those same days. '
            'Their change is the normal change. The promotion effect is what is left after we subtract that. '
            f'The fair answer is about {main["adjusted %"]:.1f}%, '
            f'and it most likely sits between {main["ci low"]:.1f}% and {main["ci high"]:.1f}%.'
        )

    st.header('What to keep in mind')
    st.write(
        'A “family” here is a whole aisle, such as all dairy, not one brand. '
        'The promotion column counts how many items were on offer. It does not say how big the discount was. '
        'Very small families can show a huge percent error and still be almost none of the store’s sales. '
        'Before mid-2015 some families were written down as zero even when they did sell, so the models use the later, reliable years.'
    )


if page.endswith('Stock the shelf'):
    page_heading('Stock the shelf', 'Type a store, tap an aisle, pick a held-out week', 'stock_shelf.jpg')
    st.write(
        'Type the store number, tap an aisle, and pick a week. '
        'The model fills the next 7 days. Only the four held-out test weeks '
        '(19 July to 15 August 2017) and the week after the records stop are listed. '
        'Weeks before 19 July 2017 were used to train the model, so a percent error on those weeks '
        'would look better than a real forecast.'
    )

    try:
        model, meta = load_live()
    except Exception as exc:
        st.error('The trained model file is missing. From the project folder run: py website/train_live_model.py')
        st.caption(str(exc))
        st.stop()
    if 'city_codes' not in meta:
        st.error(
            'The saved forecast is the old simplified model. '
            'From the project folder run: py website/train_live_model.py'
        )
        st.stop()

    history = sales_history()
    known_families = sorted(history['family'].unique())
    if 'aisle' not in st.session_state or st.session_state.aisle not in known_families:
        st.session_state.aisle = 'GROCERY I' if 'GROCERY I' in known_families else known_families[0]
    st.caption('Tap an aisle. The green button is the one on the order.')
    picked = st.columns(6)
    for i, name in enumerate(known_families):
        label = icon_for(name) + '  ' + name
        kind = 'primary' if name == st.session_state.aisle else 'secondary'
        if picked[i % 6].button(label, key='aisle-' + name, type=kind, width='stretch'):
            st.session_state.aisle = name
    family = st.session_state.aisle
    st.markdown('**On the counter:** ' + icon_for(family) + ' ' + family)

    weeks = test_weeks()
    week_labels = [label for label, _, _ in weeks]
    with st.form('forecast_form'):
        c1, c2 = st.columns(2)
        store = c1.number_input('Store number, 1 to 54', min_value=1, max_value=54, value=1, step=1)
        week_label = c2.selectbox('Week', week_labels, index=0)
        submitted = st.form_submit_button('Fill this shelf', type='primary')

    if not submitted:
        st.stop()

    week_start = {label: start for label, start, _ in weeks}[week_label]

    plan = default_plan(history, store, family, week_start)
    rows = predict_week(history, model, meta, store, family, week_start, plan)
    if rows.empty:
        st.error('Not enough past sales for that store and aisle to build a forecast.')
        st.stop()
    known = rows['sales'].notna().all()
    show = pd.DataFrame()
    show['Date'] = rows['date'].dt.strftime('%a %d %b')
    show['Forecast'] = rows['live_forecast'].to_numpy()
    show['Repeat last 4 weeks'] = rows['repeat_4_weeks'].to_numpy()
    if known:
        show.insert(1, 'Real sales', rows['sales'].to_numpy())
    show = show.set_index('Date')

    guessed = show['Forecast'].sum()
    tickets = st.columns(len(show))
    for i, (day, rec) in enumerate(show.iterrows()):
        tickets[i].markdown(
            '<div class="ticket"><div class="d">' + str(day) + '</div><div class="u">' +
            f'{rec["Forecast"]:,.0f}' + '</div><div class="d">units</div></div>',
            unsafe_allow_html=True,
        )
    if known and show['Real sales'].sum() > 0:
        total = show['Real sales'].sum()
        live_miss = 100 * (show['Forecast'] - show['Real sales']).abs().sum() / total
        simple_miss = 100 * (show['Repeat last 4 weeks'] - show['Real sales']).abs().sum() / total
        fair = (
            'This week was held out of training, so this percent is a fair test.'
            if week_start >= TEST_START
            else 'This week is inside the training period, so this percent is not a fair test.'
        )
        st.success(
            f'Store {store}, {family}. The model forecasts {guessed:,.0f} units. '
            f'The store sold {total:,.0f}. The forecast was off by {live_miss:.1f}% of the units sold. '
            f'Repeating the last 4 weeks was off by {simple_miss:.1f}%. {fair}'
        )
    elif known:
        st.info('This store sold none of this family in this week, so a percent error would not mean anything.')
    else:
        end = week_start + pd.Timedelta(days=6)
        st.success(
            f'Store {int(store)}, {family}, {week_start.strftime("%d %b")} to {end.strftime("%d %b %Y")}. '
            f'The model forecasts {guessed:,.0f} units. '
            'No sales were recorded for this week, so there is no actual number to compare.'
        )

    st.line_chart(show)
    st.caption(
        'The line is this XGBoost’s answer for the store, aisle and week you picked. '
        f'It is the Phase 2 model (52 clues, 502 trees). '
        f'On the four held-out weeks it missed {float(meta["test_wape"]):.1f}% of units, '
        'the same score as the scoreboard.'
    )


# ---------------- models ----------------
if page.endswith('Scoreboard'):
    page_heading('Scoreboard', 'XGBoost against the simple guesses', 'score_vs.jpg')
    st.write(
        'Every method was scored on the same 4 hidden weeks, one week at a time. '
        'Training stopped on 18 July 2017. The models never saw the test days while they were learning. '
        'Stock the shelf uses this same XGBoost, not a second model.'
    )

    summary = load_table('eval_summary.csv').set_index('model')
    summary = summary.rename(index=MODEL_NAMES).sort_values('WAPE %')
    scores = summary[['WAPE %', 'RMSE', 'R2']].copy()
    scores.columns = ['Missed % of units', 'Typical miss (units)', 'Fit from 0 to 1']

    best_name = scores.index[0]
    best_wape = scores.iloc[0]['Missed % of units']
    simple = scores.loc['Repeat the last 4 weeks', 'Missed % of units']
    st.success(
        f'{best_name} won. It missed {best_wape:.1f}% of units. '
        f'The best boring guess, repeating the last 4 weeks, missed {simple:.1f}%.'
    )

    st.bar_chart(scores['Missed % of units'])
    st.caption('Shorter bar is better. This is the average over all 4 test weeks, all stores, and all families.')
    st.dataframe(scores.round(3), width="stretch")

    st.subheader('Did the winner stay ahead every week?')
    by_block = load_table('eval_by_block.csv')
    weekly = by_block.pivot(index='block', columns='model', values='WAPE %')
    weekly = friendly_models(weekly)
    weekly.index = ['Week ' + str(int(i)) for i in weekly.index]
    st.line_chart(weekly)
    st.caption('Each point is that week’s miss, as a percent of units sold. XGBoost stays the lowest line.')

    st.subheader('Where the best model is strong, and where it is weak')
    st.write(
        'Big families are most of the money, so a small miss there matters. '
        'A tiny family can look terrible in percent and still be almost none of total sales.'
    )
    by_family = load_table('eval_error_by_family.csv').set_index('family')
    by_family = by_family.sort_values('share of sales %', ascending=False)
    view = by_family.rename(columns={'WAPE %': 'Missed % of units', 'share of sales %': 'Share of all sales %'})
    big = view[view['Share of all sales %'] >= 2]
    st.bar_chart(big['Missed % of units'])
    st.caption('Only families that are at least 2% of sales. PRODUCE, DAIRY, and GROCERY I are the closest.')
    st.dataframe(view.round(1), width="stretch")

    show_image(os.path.join(out, 'figures_eval', 'fig2_error_by_family.png'),
               'XGBoost miss by family. The big bars on the right are small families, not the aisles that matter most.')
    show_image(os.path.join(out, 'figures', 'shap_comparison.png'),
               'Which clues each model used. A longer bar means that clue moved the guess more.')


# ---------------- promotions ----------------
if page.endswith('Offers'):
    page_heading('Offers', 'What a promotion really adds', 'promo_offer.jpg')
    st.write(
        'Comparing the promo days with the days just before them makes the lift look bigger than it is. '
        'Sales were going to move anyway because of the weekday, holidays, and a chain that was growing. '
        'The fair number uses other stores with no promotion on the same days as the “what would have happened anyway”.'
    )

    sens = load_table('lift_sensitivity.csv')
    main = sens[sens['before days'] == 7].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric('Before vs during only', f'{main["naive %"]:.1f}%')
    c2.metric('Fair lift', f'{main["adjusted %"]:.1f}%')
    c3.metric('Promotions counted', f'{int(main["launches"]):,}')
    st.caption(
        f'The fair lift most likely sits between {main["ci low"]:.1f}% and {main["ci high"]:.1f}%. '
        'A promotion launch means the first day of a promotion after 7 days with none.'
    )

    st.subheader('The answer moves if we change the rule')
    st.write(
        'If “a new promotion” means only 3 quiet days before it, the lift looks much bigger. '
        'If we demand 14 quiet days, it is smaller. It stays positive in every version. '
        'We use 7 quiet days as the main answer because it is in the middle and still has thousands of promotions.'
    )
    rules = sens.copy()
    rules['before days'] = rules['before days'].astype(int).astype(str) + ' quiet days before'
    rules = rules.set_index('before days')
    rules = rules.rename(columns={
        'launches': 'Promotions counted',
        'naive %': 'Before vs during %',
        'adjusted %': 'Fair lift %',
        'ci low': 'Low end %',
        'ci high': 'High end %',
    })
    st.dataframe(rules.round(1), width="stretch")

    show_image(os.path.join(out, 'figures_promo', 'fig1_lift_by_length.png'),
               'Longer promotions add more. A one-day offer is a small bump. A week or more is a much bigger one.')

    st.subheader('Pick one aisle')
    launch = load_table('lift_by_family_launch.csv')
    weekly = load_table('lift_by_family_weekly.csv')
    families = sorted(set(launch['family']) | set(weekly['family']))
    family = st.selectbox('Product family', families)

    rows = []
    for name, data in [('New promotions', launch), ('Weeks when the offer suddenly got bigger', weekly)]:
        r = data[data['family'] == family]
        if len(r) > 0:
            r = r.iloc[0]
            rows.append({
                'Way of counting': name,
                'Promotions used': int(r['n']),
                'Before vs during %': round(r['naive_pct'], 1),
                'Fair lift %': round(r['adj_pct'], 1),
                'Low end %': round(r['adj_low'], 1),
                'High end %': round(r['adj_high'], 1),
            })
    st.dataframe(pd.DataFrame(rows).set_index('Way of counting'), width="stretch")
    st.caption('If the low end is below zero and the high end is above zero, this aisle’s lift is too noisy to call.')

    if family in ['SCHOOL AND OFFICE SUPPLIES', 'LAWN AND GARDEN']:
        st.warning('This family has few promotions and a strong season, so this number is not a safe one to act on.')

    pairs = load_table('cannibalization_all_pairs.csv')
    clear = pairs[pairs['p_bh'] < 0.05]
    n_drop = int((clear['adj_pct'] < 0).sum())
    n_up = int((clear['adj_pct'] > 0).sum())
    st.header('Does one aisle steal sales from another?')
    st.write(
        'We checked pairs of families. A negative number would mean the other family sold less while the first one was on offer. '
        '“Clear” means the change is unlikely to be luck.'
    )
    st.info(
        f'Out of {len(pairs)} pairs, {len(clear)} changes are clear. '
        f'{n_up} of those are increases. {n_drop} are drops. '
        'We did not find an aisle that clearly loses sales because another aisle is on promotion.'
    )

    promoted = st.selectbox('Family on promotion', sorted(pairs['A'].unique()))
    part = pairs[pairs['A'] == promoted].sort_values('adj_pct')
    view = pd.DataFrame()
    view['Other family'] = part['B'].values
    view['Change in its sales %'] = part['adj_pct'].round(1).values
    view['Low end %'] = part['adj_low'].round(1).values
    view['High end %'] = part['adj_high'].round(1).values
    view['Clear change'] = (part['p_bh'] < 0.05).map({True: 'yes', False: 'no'}).values
    st.dataframe(view.set_index('Other family'), width="stretch")

    chosen_drops = part[(part['p_bh'] < 0.05) & (part['adj_pct'] < 0)]
    if len(chosen_drops) == 0:
        st.caption(f'While {promoted} is on offer, no other family shows a clear drop.')
    else:
        names = ', '.join(chosen_drops['B'].astype(str))
        st.warning(f'While {promoted} is on offer, these families show a clear drop: {names}.')

    show_image(os.path.join(out, 'figures_promo', 'fig6_all_pairs_heatmap.png'),
               'Every family pair. Blue would be a drop. The picture does not show a band of losses.')
