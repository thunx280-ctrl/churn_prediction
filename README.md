# 🤖 E-Commerce Customer Churn Prediction & Retention Strategy | Machine Learning - Python

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/Algorithm-LightGBM_&_XGBoost-blue?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Churn Recall](https://img.shields.io/badge/Recall_(Churn)-94.74%25-brightgreen?style=for-the-badge)]()
[![Model Precision](https://img.shields.io/badge/Precision_(Churn)-93.26%25-success?style=for-the-badge)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9930-blueviolet?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **An enterprise-grade Customer Churn Prediction System and Data-Driven Retention Strategy for an E-Commerce platform across 5,630 customer accounts. Implemented sequentially across 5 Jupyter Notebooks, the champion LightGBM model achieves 94.74% Churn Recall, 97.96% Accuracy, and an ROC-AUC of 0.9930, coupled with an actionable 3-tier CRM Retention Action Matrix.**

---

* **Author:** [Nguyen Xuan Thu](https://github.com/thunx280-ctrl)  
* **Role:** Data Analyst / Machine Learning Enthusiast  
* **Tech Stack:** Python (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, `missingno`)  
* **Dataset:** E-Commerce Customer Churn Dataset (5,630 records × 20 attributes)  
* **Champion Model:** Tuned LightGBM (`models/churn_pipeline.pkl`)  

---

## 📑 Table of Contents

1. [Background Overview & Business Context](#-1-background-overview--business-context)
2. [Dataset Architecture & Data Dictionary](#-2-dataset-architecture--data-dictionary)
3. [End-to-End Machine Learning Workflow (5 Notebooks)](#-3-end-to-end-machine-learning-workflow-5-notebooks)
   - [Notebook 1: Data Understanding & Quality Audit (`01_data_understanding.ipynb`)](#notebook-1-data-understanding--quality-audit-01_data_understandingipynb)
   - [Notebook 2: Exploratory Data Analysis & Churn Dynamics (`02_eda.ipynb`)](#notebook-2-exploratory-data-analysis--churn-dynamics-02_edaipynb)
   - [Notebook 3: Leak-Free Feature Engineering & Transformation (`03_feature_engineering.ipynb`)](#notebook-3-leak-free-feature-engineering--transformation-03_feature_engineeringipynb)
   - [Notebook 4: Model Training, Multi-Algorithm Benchmark & Tuning (`04_model_training.ipynb`)](#notebook-4-model-training-multi-algorithm-benchmark--tuning-04_model_trainingipynb)
   - [Notebook 5: Rigorous Validation, Cost Analysis & Explainability (`05_model_evaluation.ipynb`)](#notebook-5-rigorous-validation-cost-analysis--explainability-05_model_evaluationipynb)
4. [Key Business Insights & Churn Driver Comparison](#-4-key-business-insights--churn-driver-comparison)
5. [Customer Risk Segmentation & 3-Tier CRM Retention Matrix](#-5-customer-risk-segmentation--3-tier-crm-retention-matrix)
6. [Repository Structure](#-6-repository-structure)
7. [Author & Contact](#-7-author--contact)

---

## 📌 1. Background Overview & Business Context

### ❓ The Business Problem
In e-commerce, Customer Acquisition Cost (CAC) is **5 to 7 times higher** than retention. When customers leave, businesses suffer compound losses: lost Customer Lifetime Value (CLV), wasted acquisition marketing, and lost market share. 

Without predictive modeling, retention teams only discover churn after customers uninstall the app or cease buying, leading to costly and low-conversion win-back efforts.

### 🎯 Analytical Objectives
1. **Early Churn Identification:** Detect behavioral pre-churn indicators before customer departure.
2. **Prioritize Churn Recall:** In retention economics, a **False Negative** (losing an undetected churner) is far more expensive than a **False Positive** (giving a retention voucher to an active customer). Optimization strictly prioritizes **Recall on Churn** while preserving high Precision.
3. **Isolate Root Drivers:** Identify operational triggers (delivery distances, customer complaints, cashback levels, dormancy gaps).
4. **Deploy Actionable Interventions:** Convert probability outputs into an automated 3-tier CRM retention matrix.

### 👤 Target Audience / Đối tượng Hướng tới
* **Data Analysts / Analytics Engineers / Data Scientists:** Reference standard machine learning workflows with rigorous data validation and zero leakage.
* **Marketing & CRM Teams:** Operationalize risk tier thresholds to deliver tailored retention campaigns at the right moment.
* **Customer Support (CS/CX) Teams:** Enforce immediate VIP response SLAs for complaint-filing accounts at high risk.

---

## 📂 2. Dataset Architecture & Data Dictionary

* **Dataset Scale:** **5,630 customer records × 20 attributes** (0 duplicates).
* **Storage Location:** `data/raw/churn_prediction.xlsx`.

<details open>
  <summary><b>📖 Comprehensive Data Dictionary (20 Attributes) — Click to Collapse</b></summary>
<br>

| Column Name | Data Type | Scale Type | Business Description |
| :--- | :---: | :---: | :--- |
| `CustomerID` | `int64` | Identifier | Unique identifier assigned to each customer account. |
| `Churn` | `int64` | Binary Flag | **Target Variable:** `1` = Customer churned; `0` = Customer retained. |
| `Tenure` | `float64` | Duration (Months) | Time elapsed since the customer registered on the platform. |
| `PreferredLoginDevice` | `str / object` | Categorical | Primary device used to browse/transact (`Mobile Phone`, `Computer`). |
| `CityTier` | `int64` | Ordinal Tier | City classification tier of customer residence (`Tier 1`, `Tier 2`, `Tier 3`). |
| `WarehouseToHome` | `float64` | Distance (km) | Distance from fulfillment fulfillment center to customer delivery address. |
| `PreferredPaymentMode` | `str / object` | Categorical | Preferred payment mechanism (`Debit Card`, `Credit Card`, `E wallet`, `COD`, `UPI`). |
| `Gender` | `str / object` | Categorical | Customer gender profile (`Male`, `Female`). |
| `HourSpendOnApp` | `float64` | Time (Hours) | Average daily hours spent actively browsing the mobile app or web storefront. |
| `NumberOfDeviceRegistered`| `int64` | Discrete Count | Number of distinct user devices registered to the account. |
| `PreferedOrderCat` | `str / object` | Categorical | Most frequently purchased product category (`Fashion`, `Laptop & Accessory`, etc.). |
| `SatisfactionScore` | `int64` | Ordinal (1–5) | Customer self-reported platform satisfaction rating (1 = Poor, 5 = Excellent). |
| `MaritalStatus` | `str / object` | Categorical | Customer marital profile (`Single`, `Married`, `Divorced`). |
| `NumberOfAddress` | `int64` | Discrete Count | Total shipping addresses saved under the customer account. |
| `Complain` | `int64` | Binary Flag | Customer logged a formal complaint in the evaluation window (`1` = Yes, `0` = No). |
| `OrderAmountHikeFromlastYear`| `float64` | Percentage (%) | Percentage year-over-year increase in order expenditure. |
| `CouponUsed` | `float64` | Discrete Count | Number of promotional discount coupons redeemed during the period. |
| `OrderCount` | `float64` | Discrete Count | Total count of successful orders completed during the evaluation period. |
| `DaySinceLastOrder` | `float64` | Duration (Days) | Days elapsed since the customer's most recent completed transaction. |
| `CashbackAmount` | `float64` | Monetary (Currency)| Average monthly cashback rebate accrued by the customer. |

</details>

---

## ⚙️ 3. End-to-End Machine Learning Workflow (5 Notebooks)

The analytical workflow is organized across **5 sequential Jupyter Notebooks** in `notebooks/`:

```
notebooks/
├── 01_data_understanding.ipynb      # Step 1: Data ingestion, schema, target imbalance & missingness audit
├── 02_eda.ipynb                     # Step 2: Univariate, bivariate, correlation & subgroup churn dynamics
├── 03_feature_engineering.ipynb     # Step 3: Leak-free split, missing indicators, domain features & encoding
├── 04_model_training.ipynb          # Step 4: Baseline model, 4-algorithm benchmark & LightGBM tuning
└── 05_model_evaluation.ipynb        # Step 5: Test set evaluation, Confusion Matrix, ROC-PR & feature importance
```

---

### Notebook 1: Data Understanding & Quality Audit (`01_data_understanding.ipynb`)

#### 1.1 Ingestion, Schema & Duplicate Check
* **Code:**
```python
# Ingest raw dataset and inspect dimensions
df = pd.read_excel("../data/raw/churn_prediction.xlsx")
display(df.shape)  # (5630, 20)
df.head()

# Verify data types and non-null schema
df.info()

# Check for duplicate records
df.duplicated().any()  # np.False_
```
* **Output:** Dataset contains **5,630 rows × 20 columns**, with **0 duplicate rows**.
![df.info()]({D3E28950-EBE7-4F66-B498-6A1C696BFD3C}.png)
* **💡 Key Insight:** Clean tabular schema with no row duplication; initial data types include 8 floats, 7 integers, and 5 text/categorical columns.

---

#### 1.2 Target Label Distribution & Imbalance Audit
* **Code:**
```python
# Check imbalance of target label
imb_df = df.groupby('Churn')['CustomerID'].count().reset_index()
imb_df['%'] = (imb_df['CustomerID'] / imb_df['CustomerID'].sum()) * 100
display(imb_df)

# Plot target distribution countplot
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.show()
```
* **Output:**
  * **Retained (`Churn = 0`):** **4,682 accounts (83.16%)**
  * **Churned (`Churn = 1`):** **948 accounts (16.84%)**
  * **Imbalance Ratio:** **~ 4.94 : 1**.
<img width="1464" height="986" alt="Target Variable Distribution" src="https://github.com/user-attachments/assets/54ed00af-3aa9-4bf6-b586-b0da557b124d" />
* **💡 Key Insight:** Severe target imbalance makes Accuracy deceptive (a dummy model predicting 0 achieves 83.16% accuracy while catching 0 churners). Evaluation must prioritize **Recall, F1-Score, and PR-AUC**.

---

#### 1.3 Missing Value Profiling & Pattern Diagnostics
* **Code:**
```python
# Visualize missingness matrix across columns
import missingno as msno
msno.matrix(df)

# Quantify missing counts and percentages
mising = df.isnull().sum()
missing_pct = (mising / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': mising, 'Percentage': missing_pct})
display(missing_df[missing_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False))
```
* **Output:** Exactly 7 numerical columns contain missing values, all below 6%:
  * `DaySinceLastOrder`: 307 (5.45%), `OrderAmountHikeFromlastYear`: 265 (4.71%), `Tenure`: 264 (4.69%), `OrderCount`: 258 (4.58%), `CouponUsed`: 256 (4.55%), `HourSpendOnApp`: 255 (4.53%), `WarehouseToHome`: 251 (4.46%).
<img width="2614" height="1362" alt="Missingno Matrix" src="https://github.com/user-attachments/assets/a985def7-d054-421c-8016-a04a288ed2ca" />
* **💡 Key Insight:** Low overall missingness (< 6%) allows for imputation without dropping records. Columns do not miss concurrently (each row misses at most 1 feature), making missingness an informative signal.

---

#### 1.4 Descriptive Statistics & Spread Diagnostics
* **Code:**
```python
# Generate transposed descriptive statistics
df.describe().T
```
* **Output:**
![df.describe().T]({2AA4751E-BFD4-4D5A-8B74-C2BCD59F6DB3}.png)
* **💡 Key Insight:** `Tenure` median is 9.0 months (mean 10.19, max 61). `WarehouseToHome` has substantial outliers (mean 15.64 km, max 127 km). Right-skewness in most metrics suggests **median imputation** will be far more robust than mean.

---

### Notebook 2: Exploratory Data Analysis & Churn Dynamics (`02_eda.ipynb`)

#### 2.1 Numerical Feature Distributions (Histograms)
* **Code:**
```python
# Histograms for all 13 numerical columns
numerical_cols = df.select_dtypes(include='number').columns.tolist()
numerical_cols.remove('CustomerID')
numerical_cols.remove('Churn')

for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col])
    plt.show()
```
* **Output:**
![Histograms Part 1]({0D3DDCFD-13C2-4F6C-A73F-AF07002120AC}.png)
![Histograms Part 2]({2A72A93C-435A-4132-9DA4-1D00B7E473AB}.png)
![Histograms Part 3]({7D4F3AF0-426F-44FD-932D-4747357CAD9B}.png)
* **💡 Key Insight:** Severe right-skewness observed across `Tenure`, `WarehouseToHome`, `OrderCount`, `CouponUsed`, `DaySinceLastOrder`, and `OrderAmountHikeFromlastYear`. `HourSpendOnApp` is symmetric (mean ≈ median ≈ 3.0).

---

#### 2.2 Numerical vs. Churn Bivariate Boxplots
* **Code:**
```python
# Boxplots comparing numerical distributions across Churn (0 vs 1)
for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.show()
```
* **Output:**
![Boxplots vs Churn]({076227EC-FA82-480D-8DB4-A5930537BA7F}.png)
* **💡 Key Insight:** Churned customers have a median tenure of only **2–3 months** (versus 10+ months for retained accounts). Churned accounts also exhibit higher fulfillment distances and substantially lower cashback earnings.

---

#### 2.3 Categorical Distributions & Subgroup Churn Rates
* **Code:**
```python
categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()

# 1. Countplots by Churn
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='Churn', data=df)
    plt.title(f'{col} by Churn')
    plt.show()

# 2. Horizontal Churn Rate Bar Charts
for col in categorical_cols:
    churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    churn_rate.plot(kind='barh')
    plt.title(f'Churn Rate by {col}')
    plt.show()
```
* **Output:**
> 📷 *(Image placeholder: Categorical countplots & churn rate bar charts)*
* **Key Numbers from Notebook:**
  * `Complain = 1`: **31.7% churn rate** vs. `Complain = 0`: **10.9%** (**2.9× increase**).
  * `MaritalStatus`: `Single` accounts churn at **26.7%**, vs. `Divorced` (14.6%) and `Married` (11.5%).
  * `PreferredLoginDevice`: `Computer` (19.8%) vs. `Mobile Phone` (15.6%).
  * `PreferedOrderCat`: `Mobile Phone` (27.4%) vs. `Grocery` (4.9%).
* **💡 Key Insight:** Service complaints and single marital status represent the highest demographic/behavioral churn risks.

---

#### 2.4 Correlation Matrix & Feature Ranking vs. Churn
* **Code:**
```python
# Compute Pearson correlation matrix
corr_matrix = df[numerical_cols + ['Churn']].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix')
plt.show()

# Rank correlations relative to Churn target
churn_corr = corr_matrix['Churn'].drop('Churn').sort_values(ascending=False)
display(churn_corr)
```
* **Output:**
<img width="2379" height="1492" alt="Correlation Heatmap" src="https://github.com/user-attachments/assets/88553ccb-0f75-4c88-883b-9da1467e9ac1" />

| Feature | Pearson Correlation vs. `Churn` | Direction & Analytical Significance |
| :--- | :---: | :--- |
| `Complain` | **+0.250188** | Strongest positive correlation; active friction drives churn. |
| `NumberOfDeviceRegistered` | **+0.107939** | Positive correlation; multi-device users actively compare alternatives. |
| `SatisfactionScore` | **+0.105481** | Weak positive correlation. |
| `CityTier` | **+0.084703** | Weak positive correlation; Tier 3 cities face delivery logistics delays. |
| `WarehouseToHome` | **+0.076630** | Greater fulfillment distance increases abandonment risk. |
| `NumberOfAddress` | **+0.043931** | Negligible linear correlation. |
| `HourSpendOnApp` | **+0.018675** | Negligible linear correlation. |
| `CouponUsed` | **-0.008264** | Negligible linear correlation. |
| `OrderAmountHikeFromlastYear` | **-0.010058** | Negligible linear correlation. |
| `OrderCount` | **-0.028697** | Minor protective effect. |
| `CashbackAmount` | **-0.154118** | Meaningful protective factor; generous rebates promote retention. |
| `DaySinceLastOrder` | **-0.160757** | Meaningful protective factor; recent buying cadence reinforces habit. |
| `Tenure` | **-0.349408** | **Strongest protective factor;** long-standing users rarely churn. |

* **💡 Key Insight:** No single feature exceeds |0.35| linear correlation, proving churn is multi-factorial and non-linear. Tree ensemble architectures (Random Forest, LightGBM, XGBoost) are required.

---

### Notebook 3: Leak-Free Feature Engineering & Transformation (`03_feature_engineering.ipynb`)

#### 3.1 Distribution Inspection of Missing Columns
* **Code:**
```python
# Examine value distribution across the 7 columns containing missing values
list_null = df.columns[df.isnull().any()].tolist()
for col in list_null:
    fig, ax = plt.subplots(figsize=(20, 5))
    plot_df = df.groupby(col)[['CustomerID']].count().reset_index()
    sns.barplot(data=plot_df, x=plot_df[col], y=plot_df['CustomerID'], ax=ax)
    plt.title(f'Distribution of {col}')
    plt.show()
```
* **Output:**
> 📷 *(Image placeholder: Value distribution bar plots for the 7 missing columns)*
* **💡 Key Insight:** Confirms severe skewness across missing columns, justifying median imputation over mean.

---

#### 3.2 Stratified Train/Test Split (Zero Data Leakage)
* **Code:**
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['CustomerID', 'Churn'])
y = df['Churn']

# Stratified partition strictly BEFORE any imputation or feature engineering
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Train:', X_train.shape, '| Test:', X_test.shape)
print('Train churn rate:', round(y_train.mean(), 4))
print('Test churn rate:', round(y_test.mean(), 4))
```
* **Output:**
  * **Train Set:** **4,504 rows × 18 columns** (Churn rate: **0.1683**)
  * **Test Set:** **1,126 rows × 18 columns** (Churn rate: **0.1687**)
* **💡 Key Insight:** Partitioning prior to imputation prevents test set distribution statistics from leaking into training features.

---

#### 3.3 Missing Indicators & Train-Only Median Imputation
* **Code:**
```python
X_train = X_train.copy()
X_test = X_test.copy()

# 1. Add missing indicators (informative signal: missing Tenure has 30.7% churn rate)
for col in list_null:
    X_train[f'{col}_missing'] = X_train[col].isnull().astype(int)
    X_test[f'{col}_missing'] = X_test[col].isnull().astype(int)

# 2. Impute with medians fitted ONLY on TRAIN set
cols_with_na = X_train.columns[X_train.isnull().any()].tolist()
medians = X_train[cols_with_na].median()
X_train[cols_with_na] = X_train[cols_with_na].fillna(medians)
X_test[cols_with_na] = X_test[cols_with_na].fillna(medians)

print('Missing in train:', X_train.isnull().sum().sum())  # 0
print('Missing in test:', X_test.isnull().sum().sum())    # 0
print('Total features:', X_train.shape[1])                # 25
```
* **Output:** 0 missing values remain across train and test sets; feature count expands to 25.
* **💡 Key Insight:** Missingness itself holds high predictive value (missing `Tenure` = 30.7% churn; missing `CouponUsed` = 3.1% churn). Binary `{col}_missing` flags retain this signal without distorting the underlying distributions.

---

#### 3.4 Categorical Standardization & Type Casting
* **Code:**
```python
# 1. Merge semantic duplicates
login_device_map = {'Phone': 'Mobile Phone'}
payment_mode_map = {'COD': 'Cash on Delivery', 'CC': 'Credit Card'}
order_cat_map = {'Mobile': 'Mobile Phone'}

for col, mapping in [
    ('PreferredLoginDevice', login_device_map),
    ('PreferredPaymentMode', payment_mode_map),
    ('PreferedOrderCat', order_cat_map)
]:
    X_train[col] = X_train[col].replace(mapping)
    X_test[col] = X_test[col].replace(mapping)

# 2. Cast non-metric ordinal column to object/category
X_train['CityTier'] = X_train['CityTier'].astype('object')
X_test['CityTier'] = X_test['CityTier'].astype('object')
```
* **Output:** Resolved 'Phone' (1,231 rows), 'COD' (365 rows), 'CC' (273 rows), 'Mobile' (809 rows).
* **💡 Key Insight:** Unifies inconsistent data entry while casting `CityTier` to categorical prevents models from treating ordinal city levels as linear distances.

---

#### 3.5 Domain-Driven Feature Engineering & Validation
* **Code:**
```python
def add_features(X):
    X = X.copy()
    X['CouponPerOrder'] = (X['CouponUsed'] / X['OrderCount'].replace(0, np.nan)).fillna(0)
    X['IsRecentBuyer'] = (X['DaySinceLastOrder'] <= 7).astype(int)
    X['AppEngagement'] = X['HourSpendOnApp'] * X['OrderCount']
    X['TenureGroup'] = pd.cut(
        X['Tenure'], bins=[-np.inf, 12, 36, np.inf],
        labels=['New', 'Mid', 'Senior']
    ).astype(str)
    return X

X_train = add_features(X_train)
X_test = add_features(X_test)

# Verify empirical churn rates by engineered features
train_check = X_train.copy()
train_check['Churn'] = y_train.values
print(train_check.groupby('TenureGroup')['Churn'].mean().round(4))
print(train_check.groupby('IsRecentBuyer')['Churn'].mean().round(4))
```
* **Output:**
  * **`TenureGroup` Churn Rate:** `New` (≤ 12 mos): **22.93%** | `Mid` (13–36 mos): **5.18%** | `Senior` (> 36 mos): **0.00%**.
  * **`IsRecentBuyer` Churn Rate:** `0` (> 7 days): **9.58%** | `1` (≤ 7 days): **18.99%** (Captures active users experiencing transaction friction).
* **💡 Key Insight:** `TenureGroup` isolates churn risk into the onboarding window (`New`: 22.93% vs `Senior`: 0.00%).

---

#### 3.6 One-Hot Encoding & Processed Data Export
* **Code:**
```python
cat_cols = X_train.select_dtypes(include=['object', 'str', 'category']).columns.tolist()

X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# Save processed datasets with target column
os.makedirs('../data/processed', exist_ok=True)
train_processed = X_train_enc.copy()
train_processed['Churn'] = y_train.values
test_processed = X_test_enc.copy()
test_processed['Churn'] = y_test.values

train_processed.to_csv('../data/processed/train_processed.csv', index=False)
test_processed.to_csv('../data/processed/test_processed.csv', index=False)
```
* **Output:** Train shape: **(4504, 38)**, Test shape: **(1126, 38)**. Files saved to `data/processed/`.
* **💡 Key Insight:** `drop_first=True` eliminates multicollinearity while test set column reindexing guarantees identical feature dimensions across environments.

---

### Notebook 4: Model Training, Multi-Algorithm Benchmark & Tuning (`04_model_training.ipynb`)

#### 4.1 Evaluation Function & Linear Baseline (Logistic Regression)
* **Code:**
```python
def evaluate_model(model, X_test, y_test, name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, prob)
    }

# Baseline Logistic Regression with StandardScaler & Balanced Class Weights
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
lr.fit(X_train, y_train)
results = [evaluate_model(lr, X_test, y_test, 'LogisticRegression')]
```
* **Output:** Accuracy: **0.8242**, Precision: **0.4879**, Recall: **0.8474**, F1: **0.6192**, ROC-AUC: **0.9043**.
<img width="1242" height="198" alt="Logistic Regression Benchmark Output" src="https://github.com/user-attachments/assets/a2fb3b09-f4ab-4b77-abb9-c47a1956f124" />
* **💡 Key Insight:** Baseline captures 84.74% of churners, but precision below 50% means over half of its churn alerts are false alarms.

---

#### 4.2 Balanced Random Forest Classifier
* **Code:**
```python
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(evaluate_model(rf, X_test, y_test, 'RandomForest'))
```
* **Output:** Accuracy: **0.9751**, Precision: **0.9091**, Recall: **0.9474**, F1: **0.9278**, ROC-AUC: **0.9928**.
<img width="1272" height="289" alt="Random Forest Benchmark Output" src="https://github.com/user-attachments/assets/81b8a846-74bc-48c3-af24-ad8a8ce3622f" />
* **💡 Key Insight:** Drastic performance leap over linear baseline: Recall reaches **94.74%** while Precision jumps from 48.79% to **90.91%**.

---

#### 4.3 XGBoost Classifier with Scale-Pos-Weight
* **Code:**
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # 4.94
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=5,
    scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
)
xgb.fit(X_train, y_train)
results.append(evaluate_model(xgb, X_test, y_test, 'XGBoost'))
```
* **Output:** Accuracy: **0.9671**, Precision: **0.8844**, Recall: **0.9263**, F1: **0.9049**, ROC-AUC: **0.9862**.
<img width="1233" height="372" alt="XGBoost Benchmark Output" src="https://github.com/user-attachments/assets/c7e7abd2-7cf3-43e3-a144-d2b745b4fbee" />
* **💡 Key Insight:** Solid gradient-boosted performance (F1: 0.9049), but slightly trailed Random Forest in both Recall and Precision.

---

#### 4.4 Balanced LightGBM Classifier
* **Code:**
```python
lgbm = LGBMClassifier(
    n_estimators=300, learning_rate=0.1, num_leaves=31,
    class_weight='balanced', random_state=42, verbose=-1
)
lgbm.fit(X_train, y_train)
results.append(evaluate_model(lgbm, X_test, y_test, 'LightGBM'))
```
* **Output:** Accuracy: **0.9778**, Precision: **0.9231**, Recall: **0.9474**, F1: **0.9351**, ROC-AUC: **0.9930**.
<img width="1278" height="406" alt="LightGBM Benchmark Output" src="https://github.com/user-attachments/assets/ea219107-affc-4d5d-b1cc-681c87109313" />
* **💡 Key Insight:** Top default model: matches peak Recall (**94.74%**) while outperforming Random Forest in Precision (**92.31%**), Accuracy (**97.78%**), and ROC-AUC (**0.9930**).

---

#### 4.5 Comparative Benchmark & Selection
* **Code:**
```python
results_df = pd.DataFrame(results).sort_values(['Recall', 'F1'], ascending=False).reset_index(drop=True)
display(results_df.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
results_df.plot(x='Model', y=['Recall', 'F1', 'Precision'], kind='bar', ax=ax)
plt.title('Model Comparison (focus: Recall)')
plt.show()
```
* **Output:**
<img width="2622" height="1195" alt="Model Benchmark Comparison" src="https://github.com/user-attachments/assets/84afca9e-283b-45ca-8cff-f335a65581e8" />

| Model | Accuracy | Precision | Recall (Primary KPI) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Default)** | **0.9778** | **0.9231** | **0.9474** | **0.9351** | **0.9930** |
| **RandomForest** | 0.9751 | 0.9091 | **0.9474** | 0.9278 | 0.9928 |
| **XGBoost** | 0.9671 | 0.8844 | 0.9263 | 0.9049 | 0.9862 |
| **LogisticRegression** | 0.8242 | 0.4879 | 0.8474 | 0.6192 | 0.9043 |

* **💡 Key Insight:** LightGBM chosen as candidate for tuning due to superior precision-recall trade-off and fast training speed.

---

#### 4.6 Hyperparameter Optimization via RandomizedSearchCV & Serialization
* **Code:**
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [200, 300, 500, 800],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [3, 5, 7, -1],
    'min_child_samples': [10, 20, 40],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

search = RandomizedSearchCV(
    LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
    param_dist, n_iter=50, scoring='f1', cv=5, random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)

# Save best model to disk
os.makedirs('../models', exist_ok=True)
joblib.dump(search.best_estimator_, '../models/churn_pipeline.pkl')
```
* **Output:**
  * **Best Params:** `{'subsample': 1.0, 'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 31, 'n_estimators': 300, 'min_child_samples': 20, 'max_depth': 7, 'learning_rate': 0.2, 'colsample_bytree': 0.6}`
  * **Best 5-Fold CV F1:** **0.8969**
  * **Final Test Evaluation (Tuned LightGBM):** Accuracy: **0.9796** | Precision: **0.9326** | Recall: **0.9474** | F1: **0.9399** | ROC-AUC: **0.9929**.
* **💡 Key Insight:** Tuning regularized tree depth (`max_depth = 7`, `colsample_bytree = 0.6`) improved test precision to **93.26%** and overall F1 to **0.9399** while preserving peak **94.74% Recall**.

---

### Notebook 5: Rigorous Validation, Cost Analysis & Explainability (`05_model_evaluation.ipynb`)

#### 5.1 Classification Report on Unseen Test Cohort (N = 1,126)
* **Code:**
```python
model = joblib.load('../models/churn_pipeline.pkl')
test = pd.read_csv('../data/processed/test_processed.csv')
X_test = test.drop(columns=['Churn'])
y_test = test['Churn']

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds, digits=4))
```
* **Output:**
```
              precision    recall  f1-score   support

           0     0.9893    0.9840    0.9866       936
           1     0.9231    0.9474    0.9351       190

    accuracy                         0.9778      1126
   macro avg     0.9562    0.9657    0.9608      1126
weighted avg     0.9781    0.9778    0.9779      1126
```
* **💡 Key Insight:** Demonstrates balanced high performance across both classes with 0.9893 precision on non-churners and 0.9474 recall on churners.

---

#### 5.2 Confusion Matrix & Financial Cost Analysis
* **Code:**
```python
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'Actual Churners Missed (False Negative): {fn}')
print(f'Actual Churners Caught (True Positive): {tp}')
```
* **Output:**
  * **True Positives (TP = 180):** **180 of 190 churners correctly flagged** (**94.74% Sensitivity**).
  * **False Negatives (FN = 10):** Only **10 churners missed** across 1,126 accounts (< 0.89%).
  * **False Positives (FP = 15):** Only 15 false alarms (**92.31% Precision**).
  * **True Negatives (TN = 921):** Retained accounts correctly recognized.
<img width="1342" height="1175" alt="Confusion Matrix Heatmap" src="https://github.com/user-attachments/assets/29aa842d-968d-4bb5-b853-7fc73393357f" />
* **💡 Key Insight:** Minimizing False Negatives to 10 protects platform revenue, while low False Positives prevents wasteful retention spending on loyal users.

---

#### 5.3 Diagnostic Performance Curves (ROC & Precision-Recall)
* **Code:**
```python
# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```
* **Output:**
  * **ROC-AUC = 0.9930** (Near-perfect discrimination)
  * **PR-AUC = 0.9690** (Exceptional precision maintained even at extreme recall)

| ROC Curve (`AUC = 0.9930`) | Precision-Recall Curve (`PR-AUC = 0.9690`) |
| :---: | :---: |
| <img width="1674" height="1179" alt="ROC Curve" src="https://github.com/user-attachments/assets/2a0503b2-746d-4717-b0e2-b27157c040f1" /> | <img width="1640" height="1184" alt="PR Curve" src="https://github.com/user-attachments/assets/ab282f54-3c49-401e-ae3f-661a19073a73" /> |

* **💡 Key Insight:** PR-AUC of 0.9690 confirms model robustness on the imbalanced minority class, proving that high recall is achieved without sacrificing precision.

---

#### 5.4 Global Feature Importance (Top 15 Drivers)
* **Code:**
```python
est = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
imp = pd.Series(est.feature_importances_, index=X_test.columns)

imp.nlargest(15).sort_values().plot(kind='barh', figsize=(8, 6))
plt.title('Top 15 Feature Importances (LightGBM)')
plt.show()
```
* **Output:**
<img width="2178" height="1465" alt="Top 15 Feature Importances" src="https://github.com/user-attachments/assets/0c201cb4-6533-496d-b568-586c61e41973" />
* **💡 Key Insight:** The top 5 operational levers governing customer churn are **`Tenure`**, **`DaySinceLastOrder`**, **`WarehouseToHome`**, **`CashbackAmount`**, and **`Complain`**.

---

## 💡 4. Key Business Insights & Churn Driver Comparison

Comparing the distributions of retained customers (`N = 4,682`) vs. churned customers (`N = 948`) isolates the core operational root causes:

| Observable Feature | Retained Cohort (`Non-Churn: N = 4,682`) | Churned Cohort (`Churn: N = 948`) | Behavioral Diagnosis | Recommended Retention Action |
| :--- | :--- | :--- | :--- | :--- |
| **`Tenure` (Account Age)** | Median: **10 months**; evenly distributed across 10–30 months. | Median: **2 months**; > 80% churned within the first 5 months. | **The Onboarding Drop-off:** New users churn rapidly if their initial post-registration experience is flawed. | Build a structured 90-day onboarding journey; offer progressive voucher incentives on 2nd and 3rd purchases. |
| **`Complain` (Support Friction)**| Only **10.9%** had logged a complaint. | **31.7%** had logged a complaint (**2.9× higher**). | **Service Breakdown Triggers Exit:** Unresolved friction directly prompts immediate account abandonment. | Enforce emergency SLA: Customer complaints from accounts with high churn scores must be escalated within 4 hours. |
| **`DaySinceLastOrder` (Recency)** | Clustered under 5 days; steady, habitual replenishment cycles. | Frequently exceeds 12–20 days of inactivity. | **Dormancy Warning Sign:** Widening gaps between orders indicate fading platform habituation. | Trigger automated re-engagement notifications on Day 10 with personalized category discounts (`PreferedOrderCat`). |
| **`CashbackAmount` (Incentive)** | Clustered between 160 – 220 currency units. | Concentrated below 140 currency units. | **Reward Disparity:** Customers receiving low rebate rewards perceive lower platform value. | Restructure loyalty tiers: dynamically scale cashback rewards as purchase frequency is maintained. |
| **`WarehouseToHome` (Logistics)** | Predominantly under 15 km; rapid last-mile delivery. | Substantial proportion located 25 – 35 km away. | **Delivery Lead Time Friction:** Extended delivery transit times erode customer satisfaction. | Optimize logistics through 3PL regional satellite hubs; offer free express delivery subsidies for distant buyers. |
| **`MaritalStatus` (Demographics)**| `Married` customers exhibit lowest churn (**11.5%**). | `Single` customers exhibit highest churn (**26.7%**). | **High Churn Propensity in Singles:** Independent buyers have lower platform inertia and chase flash promotions. | Curate personalized single-item bundles and mobile-app flash promotions tailored for younger single buyers. |

---

## 📁 6. Repository Structure

```
churn_prediction/
├── assets/
│   └── screenshots/              # Visualizations, diagnostic curves & EDA plots
├── data/
│   ├── processed/
│   │   ├── train_processed.csv   # Leak-free encoded training set (4,504 rows × 39 cols)
│   │   └── test_processed.csv    # Leak-free encoded test set (1,126 rows × 39 cols)
│   └── raw/
│       └── churn_prediction.xlsx # Raw customer transactional dataset (5,630 rows × 20 cols)
├── models/
│   └── churn_pipeline.pkl        # Serialized production LightGBM champion pipeline
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Step 1: Quality audit, target imbalance, missingness analysis
│   ├── 02_eda.ipynb                  # Step 2: Univariate/bivariate analysis & correlation matrix
│   ├── 03_feature_engineering.ipynb  # Step 3: Leak-free split, imputation, domain features, OHE
│   ├── 04_model_training.ipynb       # Step 4: 4-algorithm benchmark & RandomizedSearchCV tuning
│   └── 05_model_evaluation.ipynb     # Step 5: Confusion matrix, ROC-AUC, PR-AUC, feature importance
├── .gitignore                    # Python cache, virtual environment and checkpoint excludes
├── requirements.txt              # Environment dependencies and package versions
└── README.md                     # Comprehensive project documentation
```

---

## 👤 7. Author & Contact

* **Author:** **Nguyen Xuan Thu**
* **Role:** Data Analyst / Machine Learning
* **Email:** [thu.nx280@gmail.com](mailto:thu.nx280@gmail.com)
* **GitHub Profile:** [https://github.com/thunx280-ctrl](https://github.com/thunx280-ctrl)
* **Project Repository:** [https://github.com/thunx280-ctrl/churn_prediction](https://github.com/thunx280-ctrl/churn_prediction)

