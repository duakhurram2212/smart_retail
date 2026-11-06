features_engineered.parquet: Data Dictionary and Usage Notes

This file contains the final dataset used for predicting supermarket sales. It has 3,000,888 rows and 62 columns. The data covers 1 January 2013 to 15 August 2017. Each row represents one date, one store, and one product family. The target column is sales, which the model predicts.

To load the file, use pandas and read the Parquet file.

df = pd.read_parquet("features_engineered.parquet")

Design Rules

The forecast horizon is 7 days, which means the model predicts sales for the next 7 days. All features based on past sales, transactions, and oil prices use data that is at least 7 days old. This prevents the model from using future information.

Promotion, holiday, and calendar information can be used for the same day because it is known in advance.

25 December is missing from the original data because stores were closed. Lag features use actual calendar dates, so missing dates do not shift the values.

Features for the Kaggle test period from 16 to 31 August 2017 are not included.

Column Details

1. Keys and Target

The id column contains the original Kaggle row ID. The date, store_nbr, and family columns identify each row. The sales column contains the number of units sold and is the value we want to predict. Using log1p(sales) can help reduce the effect of very large sales values.

2. Store and Promotion Information

The onpromotion column shows how many items were on promotion that day. The city, state, store_type, and store_cluster columns provide information about each store.

3. Holiday and Event Features

Holiday features show whether a date is a national, regional, or local holiday. The is_holiday_any column indicates whether any type of holiday applies. The is_event column marks special events such as Black Friday, the earthquake period, Mother's Day, and the World Cup. Events do not always mean that stores are closed.

4. Calendar Features

These features describe the date, including the day of the week, day of the month, week, month, quarter, and year. Other features identify weekends, payday, New Year, Christmas, Mother's Day, and the earthquake period. The holiday distance features show how many days remain until the next national holiday or have passed since the previous one.

5. Sales History Features

The sales_lag columns contain sales from 7, 14, 21, and 28 days earlier. The sales_same_dow_mean_4w column calculates the average of these values.

The rolling_mean columns calculate average sales over 7, 14, and 28 day periods. The rolling_std columns measure how much sales changed during those periods. These calculations end 7 days before the prediction date.

6. Promotion Features

The promo_flag column shows whether any item was on promotion. The log_onpromotion column reduces the effect of very large promotion counts. The promo_recorded column identifies dates when promotion data was being recorded.

Other features show promotions from 7 days earlier, the number of consecutive promotion days, promotion frequency over 7 and 28 days, today's promotion intensity compared with the previous 28 day average, and the number of product families and stores involved in promotions.

7. Transactions and Oil Features

These features use previous transaction counts and oil prices. They include transactions from 7 days earlier, average transactions over a window ending 7 days earlier, oil prices from 7 days earlier, and average oil prices over 28 days ending 7 days earlier.

8. Store, Family, and Quality Features

The code columns convert store type, family, city, and state into numbers. The is_perishable column identifies product families that contain perishable goods.

The is_store_open column shows whether a store had started operating. The days_since_open column shows how many days have passed since the store's first recorded sale.

The reliable_period column is 1 from 15 July 2015 onward. This helps identify the period with more reliable sales records.

Data Quality Notes

1. A recording change occurred on 1 June 2015. Before this date, sales for PRODUCE, HOME CARE, and BEVERAGES were often recorded as zero or were incomplete. We recommend training the model using rows where reliable_period == 1.

2. Promotion data was recorded from 1 April 2014. Earlier zeros mean that promotion information was not recorded, not necessarily that there was no promotion.

3. Rows where is_store_open == 0 belong to stores that had not opened yet. These rows should be considered carefully during training.

4. Missing values appear in lag and rolling features, mainly in the first 35 days and on dates affected by missing 25 December records. Tree based models such as XGBoost and LightGBM can handle missing values. Linear models may require missing values to be removed or filled.

5. About 31 percent of all rows have zero sales. In the reliable period, this is about 16 percent. Using log1p(sales) as the target reduces skewness and can help the model learn from the data.

6. The promo_intensity_vs_28d feature has some very large values. Clipping or applying a logarithm may help when using a linear model.

7. Comparing sales on promotion days with sales on non promotion days does not directly measure the effect of promotions. A fair comparison needs a baseline for each store and product family.

Evaluation

The model uses historical information from at least 7 days earlier because forecasts are refreshed every week.

We should evaluate the model using separate 7 day test periods. Using one longer test period can cause a problem because later predictions may use sales from earlier days inside the test period. Those sales would not be available when making a real forecast at the start of the week.

For a 16 day forecast period, features based on sales and transactions would need to be at least 16 days old if the entire period is predicted at once.

Cleaning Decisions

Oil prices were placed on a complete daily calendar, and missing values were filled using the previous available price. Future prices were not used.

Transactions were set to zero on closed days when sales were zero. A 7 day median was used to fill 118 recording gaps and 2 unusual values. These issues were flagged in the cleaned dataset.

Holiday information was converted into flags to avoid creating duplicate rows. Work Day entries and transferred holidays were excluded. Events were recorded separately.

Missing 25 December dates from 2013 to 2016 were not added to the dataset.

Validation

We checked that sales_lag_7d and transactions_lag_7d matched the actual values from 7 days earlier.

We also recalculated rolling_mean_7d manually for 300 randomly selected rows and found no mismatches.

The smallest lag used is 7 days. There are no duplicate rows for the same date, store, and product family. After saving and reloading the file, the number of rows, columns, and total sales remained unchanged.
