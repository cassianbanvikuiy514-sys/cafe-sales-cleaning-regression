# Cafe Sales — Data Cleaning, EDA & Regression Model

## Problem
Predict a customer's total transaction spend (Total Spent) at a cafe based on what they ordered, when, and how — using a real-world messy dataset as practice for a full data science workflow: cleaning, exploration, feature engineering, modeling, and validation.

## Dataset
- *Source:* "Dirty Cafe Sales" dataset, Kaggle
- *Size:* 10,000 rows, 8 columns (Transaction ID, Item, Quantity, Price Per Unit, Total Spent, Payment Method, Location, Transaction Date)
- *Known issues:* Dataset was intentionally corrupted for cleaning practice — contained a mix of genuine missing values (NaN) and disguised missing values represented as the literal strings "ERROR" and "UNKNOWN". Missingness ranged from ~5% (numeric columns) up to ~40% (Location).

## Approach

### 1. Data Cleaning
- Standardized all disguised missing values ("ERROR", "UNKNOWN") into proper NaN, then converted columns to correct dtypes (numeric, datetime).
- *Recovered numeric values through cross-calculation* rather than blind imputation: since Total Spent = Quantity × Price Per Unit, missing values in any one of these three columns were recovered algebraically from the other two wherever possible. This recovered the majority of missing numeric data with 100% accuracy (not estimates).
- **Recovered Item values** by building a price → item lookup table (most items sell at one consistent price point), filling missing items wherever the price uniquely identified them. Ambiguous cases (two items sharing a price) and all remaining gaps were filled as "Unknown".
- Payment Method and Location missing values were filled with "Unknown" as their own category, since no other column could reliably predict them.
- Rows with missing Transaction Date were dropped (no reliable way to recover or estimate a date).
- Final unrecoverable numeric rows (where multiple related fields were missing simultaneously) were dropped.
- *Result:* dataset cleaned from 10,000 → 9,089 rows with zero missing values, while preserving as much real data as possible through logic-based recovery instead of guesswork.

### 2. Exploratory Data Analysis (EDA)
- Histograms showed Quantity and Price Per Unit distributed as expected given the business logic (bounded, item-driven); Total Spent was right-skewed, as is typical for spend/revenue data.
- Boxplots showed no corrupted outliers — one legitimate high-value transaction sat at the natural maximum (5 units × $5 item = $25), not an error.
- Correlation analysis: Quantity (r = 0.70) and Price Per Unit (r = 0.65) were both meaningfully and independently correlated with Total Spent (near-zero correlation with each other), confirming both were useful, non-redundant predictors.
- Category balance checks flagged that Payment Method (~30%) and Location (~40%) had large "Unknown" shares post-cleaning — noted as a caveat for interpreting those features' importance in the model.

### 3. Feature Engineering
- Extracted Transaction Month, Transaction DayOfWeek, and Transaction Weekend from the raw date, since raw datetime values aren't usable by models directly and day-of-week/seasonal patterns are common drivers of retail behavior.
- One-hot encoded Item, Payment Method, and Location (with drop_first=True to avoid redundant/multicollinear columns).
- Dropped Transaction ID (a unique, non-predictive identifier) and the original Transaction Date (superseded by the extracted features).

### 4. Modeling
- Trained a *Linear Regression* model as a baseline, given the strong linear relationship uncovered in EDA (Total Spent is largely a direct function of Quantity and Price).
- 80/20 train/test split (random_state=42 for reproducibility).

### 5. Validation
- Ran 5-fold cross-validation to confirm the model's performance was stable, not a result of a lucky train/test split.

## Results
| Metric | Single Split | 5-Fold Cross-Validation |
|---|---|---|
| R² | 0.906 | 0.907 (mean) |
| MSE | 3.47 | — |
| Std Dev (R²) | — | 0.0029 |

The model explains ~90.7% of the variance in Total Spent, with very low variance across folds — confirming consistent, reliable performance rather than a fluke.

## Key Decisions & Why
- *Cross-calculation before imputation:* Where Total Spent = Quantity × Price Per Unit could recover a missing value exactly, this was done before any estimation-based approach — since it produces the true value, not a guess.
- *"Unknown" as a category, not a dropped row:* For Payment Method and Location, dropping ~30-40% of rows would have destroyed too much of the dataset. Treating missingness as its own informative category preserved data volume while still being honest about what wasn't known.
- *Linear Regression as the first model:* Given EDA showed a strong, near-linear relationship between the core features and the target, starting with the simplest interpretable model was appropriate before reaching for more complex algorithms.

## Next Steps / What I'd Improve
- Compare against other regression models (Random Forest, Gradient Boosting) to see if nonlinear models meaningfully outperform the linear baseline.
- Apply a log transform to Total Spent to address its right-skew and test whether it improves model fit.
- Deploy the trained model as an interactive web app (e.g. via Streamlit) so predictions can be tested with live input.
- Investigate whether the large "Unknown" share in Payment Method`/Location` is influencing feature importance in ways that don't reflect real customer behavior.

## How to Run
1. Open the notebook in Google Colab (or Jupyter).
2. Download the "Dirty Cafe Sales" dataset from Kaggle and place it in the working directory as dirty_cafe_sales.csv.
3. Run all cells in order — the notebook walks through cleaning → EDA → feature engineering → modeling → validation sequentially.

## Tools Used
Python, Pandas, NumPy, Matplotlib, scikit-learn (LinearRegression, train_test_split, cross_val_score)
