# Part B: Business Case Analysis

## Scenario: Promotion Effectiveness at a Fashion Retail Chain

The company operates 50 stores across urban, semi-urban, and rural locations. Each month, different promotions may be used, such as Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. The objective is to decide which promotion should be deployed in each store each month to maximise the number of items sold.

---

# B1. Problem Formulation

## (a) Machine Learning Problem

This can be formulated as a supervised machine learning regression problem.

The target variable is:

- `items_sold`

The model will predict the expected number of items sold for a given store, month, and promotion type.

Candidate input features may include:

- Store-related features: `store_id`, `store_size`, `location_type`, monthly footfall, local competition density
- Promotion-related features: `promotion_type`, discount level, promotion duration
- Calendar-related features: `month`, `season`, `is_weekend`, `is_festival`, `holiday_period`
- Customer-related features: customer demographics, average basket size, customer segment
- Historical performance features: previous month sales volume, previous promotion response, average monthly items sold

This is a regression problem because the output variable, `items_sold`, is a continuous numerical value. The business wants to predict how many items will be sold, not simply classify whether a promotion is good or bad.

---

## (b) Why Items Sold Is a Better Target Than Revenue

Using total sales revenue as the target can be misleading because revenue is affected by product prices, discounts, and product mix. For example, a store may generate high revenue by selling fewer expensive items, while another store may sell many lower-priced items and generate lower revenue. If the goal is to measure promotion effectiveness in terms of customer response, `items_sold` is more reliable because it directly captures sales volume.

Promotions often reduce prices, so revenue may fall even when the promotion successfully increases customer purchases. Therefore, using revenue alone may incorrectly make a successful discount promotion look weak.

This illustrates an important principle in real-world machine learning: the target variable should match the actual business objective. A poorly chosen target can make the model optimise the wrong outcome, even if the model is technically accurate.

---

## (c) Alternative Modelling Strategy

Instead of using one single global model across all 50 stores, a better strategy would be to use a segmented or hierarchical modelling approach.

One option is to build separate models for different store groups, such as:

- Urban stores
- Semi-urban stores
- Rural stores

Another option is to build one global model but include important store-level features such as `store_id`, `location_type`, `store_size`, footfall, and competition density. Interaction features between `promotion_type` and `location_type` can also be added.

This is better because stores in different locations may respond differently to the same promotion. For example, urban customers may respond better to loyalty points, while rural customers may respond better to flat discounts. A single global model may hide these local differences and produce average recommendations that are not optimal for every store.

---

# B2. Data and EDA Strategy

## (a) Creating a Clean Modelling Dataset

The raw data comes from four separate tables:

1. Transactions table
2. Store attributes table
3. Promotion details table
4. Calendar table

The modelling dataset should be created at the store-month-promotion level. Each row should represent one store in one month under one promotion condition.

Example row:

```text
Store 12 | December 2025 | Loyalty Points Bonus | store attributes | calendar features | items_sold
```

The transactions table should be aggregated to monthly store level. Useful aggregations include:

Total items_sold
Total revenue
Number of transactions
Average basket size
Average selling price
Number of unique customers, if available

The store attributes table should be joined using store_id. The promotion details table should be joined using promotion identifiers and dates. The calendar table should be joined using transaction date or month.

Before modelling, the final dataset should be checked for missing values, duplicate rows, inconsistent date formats, and mismatched store or promotion IDs.

## (b) EDA Before Modelling

Before building the model, the following EDA should be performed:

1. Items Sold by Promotion Type

A bar chart or box plot can show how average items_sold changes across different promotion types. This helps identify which promotions generally perform better.

If some promotions show much higher sales volume, promotion type should be treated as an important predictive feature.

2. Items Sold by Location Type

A chart comparing sales volume across urban, semi-urban, and rural stores can reveal location-based differences.

If performance differs strongly by location, the model should include location_type and possibly interaction features between promotion and location.

3. Monthly or Seasonal Sales Trend

A line chart of monthly items_sold can show seasonality, festival peaks, or year-end shopping behaviour.

If seasonality is visible, features such as month, festival_period, and holiday_period should be included.

4. Promotion Effectiveness by Store Size

A grouped bar chart can compare promotion performance across small, medium, and large stores.

This can reveal whether certain promotions work better in larger stores due to higher footfall or more product variety.

5. Correlation Analysis for Numerical Features

A correlation heatmap can show relationships between numerical variables such as footfall, competition density, basket size, and items sold.

Highly useful variables can be kept, while redundant or weak variables may be reviewed before modelling.

## (c) Handling Promotion Imbalance

If 80% of transactions occurred without promotion, the dataset is imbalanced. This can cause the model to learn the non-promotion pattern very well but perform poorly when predicting the effect of specific promotions.

The model may underestimate the impact of less frequent promotions because there are fewer examples for those cases.

To address this:

Evaluate performance separately for promotion and non-promotion cases
Use stratified sampling while creating validation sets
Create balanced training samples where possible
Add clear promotion indicator features
Collect more data for underrepresented promotions
Use weighting or careful validation to ensure minority promotion types are not ignored

The goal is to make sure the model learns the effect of each promotion type, not just the dominant no-promotion behaviour.

# B3. Model Evaluation and Deployment

## (a) Train-Test Split and Evaluation

Since the dataset contains monthly store-level data for three years across 50 stores, the train-test split should be time-based.

A suitable approach would be:

Use the first 80% of months for training
Use the most recent 20% of months for testing

For example, if there are 36 months of data, the first 28 or 29 months can be used for training and the remaining recent months can be used for testing.

A random split is inappropriate because it can cause data leakage. In a real business situation, the model is trained on past data and used to predict future performance. If random splitting is used, future months may appear in the training data, making the model look more accurate than it really is.

Suitable evaluation metrics include:

RMSE: useful because it penalises large errors more heavily
MAE: useful because it shows the average prediction error in items sold
R²: useful for understanding how much variation the model explains

For business interpretation, MAE is especially easy to explain. For example, if the MAE is 20, it means the model is wrong by around 20 items on average.

## (b) Explaining Different Recommendations for the Same Store

The model may recommend Loyalty Points Bonus for Store 12 in December and Flat Discount for Store 12 in March because the context is different in each month.

In December, customer behaviour may be influenced by festivals, holidays, gifting, and higher shopping intent. If the model finds that loyalty-related promotions perform well during festive or high-demand months, it may recommend Loyalty Points Bonus.

In March, customers may be more price-sensitive or demand may be lower. If the model finds that Flat Discount performs better during lower-demand months or end-of-season periods, it may recommend Flat Discount.

Feature importance or SHAP values can help explain these recommendations. Important factors may include:

Month
Festival indicator
Promotion type
Store location
Store size
Historical footfall
Competition density
Previous sales pattern

For the same store, the recommendation can change because time-based and calendar-based features change. The model is not only looking at the store; it is also considering the month, season, festival period, and expected customer behaviour.

## (c) End-to-End Deployment Process

After the model is trained and validated, it should be saved using a tool such as joblib or pickle. The saved object should include the full preprocessing and modelling pipeline so that the same transformations are applied during prediction.

The monthly deployment process would work as follows:

Prepare the latest monthly input data for all 50 stores.
Create possible rows for each store and each candidate promotion.
Add required store, promotion, and calendar features.
Load the saved model pipeline.
Predict expected items_sold for each store-promotion combination.
Select the promotion with the highest predicted items_sold for each store.
Share the final recommendation table with the marketing team.

The recommendation output could look like:

store_id | month | recommended_promotion | predicted_items_sold

The model does not need to be retrained every month. It can be reused for prediction as long as the input data is prepared in the same format.

After deployment, performance should be monitored by comparing predicted sales with actual sales. Important monitoring checks include:

Prediction error by month
Prediction error by store type
Promotion-level performance
Changes in customer behaviour
Data drift in features such as footfall or competition density

If model performance starts declining, the model should be retrained using newer data.