# Pandas Advanced: Complete Guide for AI/ML

You already use Pandas as a Data Engineer. This guide focuses on the **advanced techniques** you need for AI/ML workflows — feature engineering, data preprocessing, and efficient large-scale operations.

---

## 📌 1. Advanced Data Selection & Filtering

### 1.1 Multi-Condition Filtering
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 28],
    'salary': [50000, 60000, 75000, 90000, 55000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'performance': [4.2, 3.8, 4.5, 3.9, 4.1]
})

# Complex filtering
result = df.query("age > 28 and salary < 80000 and department == 'IT'")

# Using .loc with multiple conditions
mask = (df['age'] > 25) & (df['salary'] > 55000) & (df['performance'] >= 4.0)
result = df.loc[mask, ['age', 'salary', 'performance']]

# isin for categorical filtering
result = df[df['department'].isin(['IT', 'Finance'])]

# between for range filtering
result = df[df['salary'].between(50000, 75000)]
```

### 1.2 Advanced Indexing
```python
# MultiIndex (hierarchical indexing)
arrays = [['US', 'US', 'India', 'India'], ['Q1', 'Q2', 'Q1', 'Q2']]
index = pd.MultiIndex.from_arrays(arrays, names=['country', 'quarter'])
df = pd.DataFrame({'revenue': [100, 150, 80, 120], 'profit': [20, 30, 15, 25]}, index=index)

# Access multi-level
df.loc['US']           # All US data
df.loc[('US', 'Q1')]   # Specific entry
df.xs('Q1', level='quarter')  # Cross-section across all countries
```

---

## 📌 2. Feature Engineering (The Heart of ML)

### 2.1 Creating New Features
```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'sales': np.random.randint(100, 1000, 100),
    'price': np.random.uniform(10, 100, 100),
    'quantity': np.random.randint(1, 50, 100)
})

# Mathematical features
df['revenue'] = df['price'] * df['quantity']
df['log_sales'] = np.log1p(df['sales'])  # Log transform (handles skewness)
df['sales_squared'] = df['sales'] ** 2

# Binning (discretization)
df['price_category'] = pd.cut(df['price'], bins=[0, 30, 60, 100],
                               labels=['low', 'medium', 'high'])
df['sales_quartile'] = pd.qcut(df['sales'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Interaction features
df['price_x_quantity'] = df['price'] * df['quantity']
```

### 2.2 DateTime Feature Extraction
```python
# Extract time-based features (critical for time-series ML)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek      # 0=Monday
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['quarter'] = df['date'].dt.quarter
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week

# Cyclical encoding (for neural networks)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

### 2.3 Text Feature Engineering
```python
df = pd.DataFrame({
    'text': ['Hello World!', 'Machine Learning is great', 'AI Engineer role', 'Data Science']
})

# Basic text features
df['word_count'] = df['text'].str.split().str.len()
df['char_count'] = df['text'].str.len()
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
df['has_exclamation'] = df['text'].str.contains('!').astype(int)
df['upper_count'] = df['text'].str.count(r'[A-Z]')

# String operations
df['text_lower'] = df['text'].str.lower()
df['text_clean'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
```

---

## 📌 3. Handling Missing Data (ML Preprocessing)

### 3.1 Detection & Analysis
```python
df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [np.nan, 2, np.nan, 4, 5],
    'C': [1, 2, 3, 4, 5]
})

# Missing data report
missing_report = pd.DataFrame({
    'missing_count': df.isnull().sum(),
    'missing_percent': (df.isnull().sum() / len(df)) * 100,
    'dtype': df.dtypes
})
print(missing_report)
```

### 3.2 Imputation Strategies
```python
# Simple imputation
df['A_mean'] = df['A'].fillna(df['A'].mean())        # Mean imputation
df['A_median'] = df['A'].fillna(df['A'].median())     # Median imputation
df['A_mode'] = df['A'].fillna(df['A'].mode()[0])       # Mode imputation

# Forward/Backward fill (time-series)
df['A_ffill'] = df['A'].ffill()   # Forward fill
df['A_bfill'] = df['A'].bfill()   # Backward fill

# Interpolation (best for time-series)
df['A_interp'] = df['A'].interpolate(method='linear')
df['A_spline'] = df['A'].interpolate(method='spline', order=2)

# Group-based imputation (smarter)
df['department'] = ['IT', 'IT', 'HR', 'HR', 'IT']
df['A_group'] = df.groupby('department')['A'].transform(lambda x: x.fillna(x.mean()))

# Create missing indicator feature (useful for tree models)
df['A_is_missing'] = df['A'].isnull().astype(int)
```

---

## 📌 4. GroupBy & Aggregation (Advanced)

### 4.1 Multi-Column Aggregation
```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'sub_cat': ['x', 'x', 'y', 'y', 'x', 'y'],
    'sales': [100, 200, 150, 300, 120, 250],
    'profit': [10, 30, 20, 40, 15, 35]
})

# Multiple aggregations
result = df.groupby('category').agg(
    total_sales=('sales', 'sum'),
    avg_sales=('sales', 'mean'),
    max_profit=('profit', 'max'),
    count=('sales', 'count'),
    std_sales=('sales', 'std')
).reset_index()

# Custom aggregation functions
result = df.groupby('category').agg(
    sales_range=('sales', lambda x: x.max() - x.min()),
    profit_ratio=('profit', lambda x: x.sum() / x.count()),
    cv=('sales', lambda x: x.std() / x.mean())  # Coefficient of variation
)
```

### 4.2 Transform & Apply
```python
# Transform: returns same-size output (great for normalization per group)
df['sales_zscore'] = df.groupby('category')['sales'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Rank within group
df['sales_rank'] = df.groupby('category')['sales'].transform('rank', ascending=False)

# Percentage of group total
df['sales_pct'] = df.groupby('category')['sales'].transform(lambda x: x / x.sum())
```

### 4.3 Window Functions (Time-Series Features)
```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30, freq='D'),
    'sales': np.random.randint(50, 200, 30)
})

# Rolling windows (moving averages — very common in forecasting)
df['rolling_mean_7'] = df['sales'].rolling(window=7).mean()
df['rolling_std_7'] = df['sales'].rolling(window=7).std()
df['rolling_min_7'] = df['sales'].rolling(window=7).min()
df['rolling_max_7'] = df['sales'].rolling(window=7).max()

# Expanding windows (cumulative)
df['expanding_mean'] = df['sales'].expanding().mean()

# Lag features (critical for time-series ML)
df['lag_1'] = df['sales'].shift(1)      # Yesterday's sales
df['lag_7'] = df['sales'].shift(7)      # Last week's sales
df['diff_1'] = df['sales'].diff(1)      # Day-over-day change
df['pct_change'] = df['sales'].pct_change()  # Percentage change

# Exponential weighted moving average (EWMA)
df['ewm_mean'] = df['sales'].ewm(span=7).mean()
```

---

## 📌 5. Merging & Joining (Advanced)

### 5.1 Different Join Types
```python
users = pd.DataFrame({'user_id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
orders = pd.DataFrame({'user_id': [1, 1, 2, 4], 'amount': [100, 200, 150, 300]})

# All join types
inner = pd.merge(users, orders, on='user_id', how='inner')    # Only matches
left = pd.merge(users, orders, on='user_id', how='left')      # All users
right = pd.merge(users, orders, on='user_id', how='right')    # All orders
outer = pd.merge(users, orders, on='user_id', how='outer')    # Everything

# Merge with indicator (see which rows matched)
merged = pd.merge(users, orders, on='user_id', how='outer', indicator=True)
# _merge column: 'both', 'left_only', 'right_only'
```

### 5.2 Merge on Multiple Keys & Conditions
```python
# Merge on multiple columns
df1 = pd.DataFrame({'year': [2023, 2024], 'quarter': ['Q1', 'Q1'], 'revenue': [100, 120]})
df2 = pd.DataFrame({'year': [2023, 2024], 'quarter': ['Q1', 'Q1'], 'profit': [20, 25]})
merged = pd.merge(df1, df2, on=['year', 'quarter'])

# Merge nearest (for time-series alignment)
trades = pd.DataFrame({
    'time': pd.to_datetime(['2024-01-01 09:30:01', '2024-01-01 09:30:05']),
    'price': [100, 102]
})
quotes = pd.DataFrame({
    'time': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:30:03']),
    'bid': [99, 101]
})
result = pd.merge_asof(trades.sort_values('time'), quotes.sort_values('time'), on='time')
```

---

## 📌 6. Encoding Categorical Variables (ML-Ready Data)

```python
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'XL', 'M'],
    'rating': ['low', 'medium', 'high', 'medium', 'low']
})

# One-Hot Encoding (for nominal categories — no order)
one_hot = pd.get_dummies(df['color'], prefix='color', drop_first=True)

# Label Encoding (for ordinal categories — have order)
size_order = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
df['size_encoded'] = df['size'].map(size_order)

# Frequency Encoding (count-based)
freq = df['color'].value_counts(normalize=True)
df['color_freq'] = df['color'].map(freq)

# Target Encoding (mean of target per category — powerful but risk of leakage)
df['target'] = [1, 0, 1, 0, 1]
target_mean = df.groupby('color')['target'].mean()
df['color_target_enc'] = df['color'].map(target_mean)
```

---

## 📌 7. Performance Optimization

### 7.1 Memory Optimization
```python
# Check memory usage
df.info(memory_usage='deep')

# Downcast numeric types
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')

# Convert object to category (huge savings for repeated strings)
df['category_col'] = df['category_col'].astype('category')

# Optimized read
df = pd.read_csv('large_file.csv',
    usecols=['col1', 'col2'],       # Only needed columns
    dtype={'col1': 'int32', 'col2': 'category'},  # Specify types
    nrows=10000                     # Limit rows for testing
)
```

### 7.2 Chunked Processing (Large Files)
```python
# Process in chunks (when file doesn't fit in RAM)
chunk_results = []
for chunk in pd.read_csv('huge_file.csv', chunksize=100000):
    processed = chunk.groupby('category')['value'].sum()
    chunk_results.append(processed)

final = pd.concat(chunk_results).groupby(level=0).sum()
```

### 7.3 Vectorized String Operations
```python
# ❌ Slow: apply with lambda
df['result'] = df['text'].apply(lambda x: x.lower().strip())

# ✅ Fast: vectorized string methods
df['result'] = df['text'].str.lower().str.strip()

# ❌ Slow: iterrows
for idx, row in df.iterrows():
    df.at[idx, 'new_col'] = row['A'] + row['B']

# ✅ Fast: vectorized
df['new_col'] = df['A'] + df['B']
```

---

## 📌 8. Pivot Tables & Cross-Tabulation

```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=12, freq='M'),
    'product': ['A', 'B', 'C'] * 4,
    'region': ['North', 'South'] * 6,
    'sales': np.random.randint(100, 500, 12)
})

# Pivot table (like Excel but powerful)
pivot = df.pivot_table(
    values='sales',
    index='product',
    columns='region',
    aggfunc=['mean', 'sum', 'count'],
    margins=True  # Add totals
)

# Cross-tabulation (frequency tables)
ct = pd.crosstab(df['product'], df['region'], margins=True, normalize='index')
```

---

## 📌 9. Method Chaining (Clean ML Preprocessing Pipeline)

```python
# Build entire preprocessing pipeline as a chain
clean_df = (
    pd.read_csv('raw_data.csv')
    .rename(columns=str.lower)
    .rename(columns=lambda x: x.replace(' ', '_'))
    .drop(columns=['unnecessary_col'])
    .dropna(subset=['important_col'])
    .assign(
        log_value=lambda x: np.log1p(x['value']),
        category_encoded=lambda x: x['category'].map({'A': 0, 'B': 1, 'C': 2}),
        date_parsed=lambda x: pd.to_datetime(x['date']),
    )
    .query("value > 0 and category_encoded.notna()")
    .sort_values('date_parsed')
    .reset_index(drop=True)
)
```

---

## 📌 10. Pandas → ML Model Ready

### Complete Preprocessing Pipeline Example
```python
def prepare_ml_data(df, target_col, test_size=0.2):
    """Full pandas preprocessing for ML."""

    # 1. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna('Unknown')

    # 3. Encode categoricals
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 4. Normalize numerics
    X[numeric_cols] = (X[numeric_cols] - X[numeric_cols].mean()) / X[numeric_cols].std()

    # 5. Split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test
```

---

## 🎤 Interview Q&A (Top 10 Pandas Questions)

### Q1: What is the difference between `apply()`, `map()`, and `applymap()`?
> **How to answer:** Show you know which works on what level.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# map() — Series only, element-wise
df['A'].map(lambda x: x * 2)             # Series → Series

# apply() — Works on both Series and DataFrame
df['A'].apply(lambda x: x * 2)           # Series → Series
df.apply(lambda row: row['A'] + row['B'], axis=1)  # Row-wise
df.apply(lambda col: col.sum(), axis=0)   # Column-wise

# applymap() — DataFrame only, element-wise (deprecated in newer versions, use map())
df.map(lambda x: x * 2)                  # Every cell
```
**Key insight:** `apply()` with `axis=1` is slow for large DataFrames. Use vectorized operations instead.

---

### Q2: How do you handle data leakage during feature engineering?
> **How to answer:** This separates senior from junior candidates.

```python
# ❌ WRONG: Fitting on entire dataset before splitting
df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
# This leaks test data statistics into training!

# ✅ CORRECT: Fit on train, transform both
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.2)
train_mean = X_train['value'].mean()
train_std = X_train['value'].std()

X_train['normalized'] = (X_train['value'] - train_mean) / train_std
X_test['normalized'] = (X_test['value'] - train_mean) / train_std
# Using TRAIN statistics for BOTH — no leakage!

# Same applies to: Target encoding, imputation, and scaling
```

---

### Q3: What's the difference between `merge()`, `join()`, and `concat()`?
> **How to answer:** Show when to use each one.

```python
# merge() — SQL-like joins on columns
pd.merge(df1, df2, on='key', how='left')

# join() — joins on index (shorthand for merge on index)
df1.join(df2, how='inner')  # Joins on index

# concat() — stacking DataFrames (no join logic)
pd.concat([df1, df2], axis=0)  # Stack rows (UNION ALL)
pd.concat([df1, df2], axis=1)  # Stack columns side by side

# When to use:
# merge() → combining on shared columns (most common in ML)
# join()  → combining on index
# concat() → appending rows or adding new feature columns
```

---

### Q4: How do you create lag features for time-series? Why are they important?
> **How to answer:** Directly relevant for forecasting interviews.

```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10, freq='D'),
    'sales': [100, 120, 90, 150, 130, 110, 140, 160, 125, 145]
})

# Lag features — use past values as predictors
df['lag_1'] = df['sales'].shift(1)    # Yesterday
df['lag_7'] = df['sales'].shift(7)    # Last week same day

# Rolling features — capture trends
df['rolling_mean_3'] = df['sales'].rolling(3).mean()
df['rolling_std_3'] = df['sales'].rolling(3).std()

# Difference features — capture momentum
df['diff_1'] = df['sales'].diff(1)
df['pct_change'] = df['sales'].pct_change()

# ⚠️ CRITICAL: Drop NaN rows created by shift/rolling
df = df.dropna()
```
**Why they matter:** "The best predictor of tomorrow's sales is today's sales." Lag features give models temporal context.

---

### Q5: How would you handle a dataset with 50 million rows in Pandas?
> **How to answer:** Show you understand memory constraints.

```python
# Strategy 1: Read only needed columns + optimized dtypes
df = pd.read_csv('huge.csv',
    usecols=['col1', 'col2', 'col3'],
    dtype={'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
)

# Strategy 2: Chunk processing
results = []
for chunk in pd.read_csv('huge.csv', chunksize=500_000):
    processed = chunk.groupby('category')['value'].mean()
    results.append(processed)
final = pd.concat(results).groupby(level=0).mean()

# Strategy 3: Use Parquet format (columnar, compressed)
df.to_parquet('data.parquet')  # 5-10x smaller than CSV
df = pd.read_parquet('data.parquet', columns=['col1', 'col2'])

# Strategy 4: Switch to Polars or Dask for truly large data
# import polars as pl
# df = pl.read_csv('huge.csv')  # 10x faster than Pandas
```
**Bonus answer:** "As a Data Engineer, I'd suggest storing it in Parquet on S3 and using PySpark for distributed processing."

---

### Q6: Explain the difference between `transform()` and `apply()` in GroupBy.
> **How to answer:** Critical for feature engineering interviews.

```python
df = pd.DataFrame({
    'dept': ['IT', 'IT', 'HR', 'HR'],
    'salary': [80000, 90000, 60000, 70000]
})

# apply() — returns ANY shape (aggregated, can change structure)
df.groupby('dept')['salary'].apply(lambda x: x.sum())
# Returns: IT: 170000, HR: 130000 (2 rows)

# transform() — returns SAME shape as input (broadcasts back)
df['dept_mean'] = df.groupby('dept')['salary'].transform('mean')
# Returns: [85000, 85000, 65000, 65000] (4 rows — same as original!)

# Use transform for: Z-score per group, rank per group, percentage of total
df['salary_zscore'] = df.groupby('dept')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

---

### Q7: How do you detect and handle outliers?
> **How to answer:** Show multiple approaches with trade-offs.

```python
# Method 1: IQR (Interquartile Range)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
mask = (df['value'] >= Q1 - 1.5*IQR) & (df['value'] <= Q3 + 1.5*IQR)
df_clean = df[mask]

# Method 2: Z-score (for normally distributed data)
z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
df_clean = df[z_scores.abs() < 3]  # Keep within 3 std

# Method 3: Clipping (don't remove, just cap)
df['value_clipped'] = df['value'].clip(
    lower=df['value'].quantile(0.01),
    upper=df['value'].quantile(0.99)
)

# When to use which:
# IQR    → general purpose, robust to skew
# Z-score → normally distributed data
# Clipping → when you can't afford to lose rows
```

---

### Q8: What is target encoding and when would you use it over one-hot?
> **How to answer:** Shows advanced feature engineering knowledge.

```python
# One-Hot: Creates N new columns for N categories
# Problem: 10,000 unique cities = 10,000 new columns! (curse of dimensionality)

# Target Encoding: Replace category with mean of target variable
train_df = pd.DataFrame({
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'Chicago'],
    'purchased': [1, 0, 1, 0, 1, 0]
})

# Calculate mean target per category
target_means = train_df.groupby('city')['purchased'].mean()
train_df['city_encoded'] = train_df['city'].map(target_means)
# NYC→1.0, LA→0.5, Chicago→0.0

# ⚠️ Danger: Data leakage! Use K-fold target encoding:
# For each fold, compute target mean on other folds only
```
**When to use:**
- One-hot: < 10 categories, tree models
- Target encoding: High cardinality (100+ categories), gradient boosting

---

### Q9: How do you handle the `SettingWithCopyWarning`?
> **How to answer:** Shows you write production-quality Pandas code.

```python
# ❌ Causes warning (and potentially incorrect results)
df_subset = df[df['age'] > 30]
df_subset['new_col'] = 1  # SettingWithCopyWarning!

# Problem: Is df_subset a view or copy? It's ambiguous.

# ✅ Fix 1: Explicit copy
df_subset = df[df['age'] > 30].copy()
df_subset['new_col'] = 1  # Safe!

# ✅ Fix 2: Use .loc on original
df.loc[df['age'] > 30, 'new_col'] = 1  # Modifies original directly

# ✅ Fix 3: Method chaining with assign
df_subset = (
    df[df['age'] > 30]
    .copy()
    .assign(new_col=1)
)
```

---

### Q10: Write a complete feature engineering pipeline for a classification task.
> **How to answer:** Combines everything — the ultimate Pandas interview question.

```python
def feature_pipeline(df, target_col, is_train=True, train_stats=None):
    """Production-grade feature engineering pipeline."""
    df = df.copy()

    # 1. DateTime features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df = df.drop(columns=['date'])

    # 2. Handle missing values
    numeric_cols = df.select_dtypes(include='number').columns.drop(target_col, errors='ignore')
    cat_cols = df.select_dtypes(include='object').columns

    if is_train:
        train_stats = {
            'medians': df[numeric_cols].median(),
            'means': df[numeric_cols].mean(),
            'stds': df[numeric_cols].std()
        }

    df[numeric_cols] = df[numeric_cols].fillna(train_stats['medians'])
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    # 3. Encode categoricals
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4. Normalize (using TRAIN stats only)
    df[numeric_cols] = (df[numeric_cols] - train_stats['means']) / train_stats['stds']

    return df, train_stats

# Usage:
# train_processed, stats = feature_pipeline(train_df, 'target', is_train=True)
# test_processed, _ = feature_pipeline(test_df, 'target', is_train=False, train_stats=stats)
```

---

## 🎯 Mastery Checklist

- [ ] Multi-condition filtering with `.query()` and boolean masks
- [ ] Feature engineering (datetime, text, mathematical)
- [ ] Missing data handling (imputation strategies)
- [ ] Advanced GroupBy (transform, window functions, lag features)
- [ ] Complex merges and joins
- [ ] Categorical encoding (one-hot, label, target, frequency)
- [ ] Memory optimization and chunked processing
- [ ] Method chaining for clean preprocessing pipelines
- [ ] Build a complete ML preprocessing pipeline in Pandas
- [ ] Answer all 10 interview questions confidently
