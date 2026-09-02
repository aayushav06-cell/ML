# Machine Learning Day 2 Tutorial Notes

Building on Day 1's foundation, today we go deeper into the algorithms and the metrics that tell us whether they're actually working.

---

## 1. Data Preprocessing

Real-world data is messy. Before any model sees it, you almost always need to clean and reshape it.

### 1.1 Handling Missing Values
- **Drop rows/columns** — only when missingness is small and random
- **Fill (impute) values**:
  - Mean / median for numeric features
  - Mode (most frequent) for categorical features
  - `KNNImputer` or `IterativeImputer` for smarter fills

```python
from sklearn.impute import SimpleImputer

# Numeric: fill with the mean
num_imputer = SimpleImputer(strategy='mean')
X_num = num_imputer.fit_transform(X_num)

# Categorical: fill with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat = cat_imputer.fit_transform(X_cat)
```

### 1.2 Encoding Categorical Variables
ML models want numbers, not strings.

- **Label Encoding** — assign each category an integer (good for ordinal data: low/med/high)
- **One-Hot Encoding** — create a binary column per category (good for nominal data: red/blue/green)

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.get_dummies(df, columns=['color'], drop_first=True)
# drop_first=True avoids the dummy-variable trap (multicollinearity)
```

### 1.3 Feature Scaling
Many algorithms (logistic regression, SVM, KNN, K-means, PCA) are distance-based — unscaled features dominate the result.

- **StandardScaler** → mean 0, std 1 (the default choice)
- **MinMaxScaler** → values in [0, 1] (good for neural nets, image data)
- **RobustScaler** → uses median/IQR, good with outliers

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Rule of thumb**: fit the scaler on training data only, then apply it to test data. Same for imputers and encoders.

### 1.4 Outlier Detection
- **IQR method**: anything below Q1 − 1.5·IQR or above Q3 + 1.5·IQR
- **Z-score**: |z| > 3 is a common cutoff
- Tree-based models are mostly insensitive to outliers; linear models are not.

### 1.5 Train / Validation / Test Split
Day 1 used train/test. For tuning hyperparameters properly, split into three:

| Set | Purpose | Typical size |
|-----|---------|--------------|
| Train | Fit the model | 60–70% |
| Validation | Tune hyperparameters, compare models | 15–20% |
| Test | Final, unbiased evaluation | 15–20% |

Or skip the validation set and use **cross-validation** (see below).

---

## 2. Model Evaluation Metrics — Deep Dive

Day 1 named the metrics. Today we learn when each one lies to you.

### 2.1 The Confusion Matrix
For binary classification, every prediction falls into one of four cells:

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN | FP (Type I error) |
| **Actual Positive** | FN (Type II error) | TP |

```python
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 2.2 When Accuracy Misleads
Consider a fraud dataset where 0.1% of transactions are fraudulent. A model that always predicts "not fraud" is **99.9% accurate** and completely useless.

- **Imbalanced classes** → look at precision, recall, F1, ROC-AUC — not accuracy.
- **Cost of errors differ** → e.g., missing a cancer (FN) is worse than a false alarm (FP).

### 2.3 Precision vs Recall — The Tradeoff
- **Precision** = "When I say positive, how often am I right?" — important when FP is expensive (spam filter flagging real email).
- **Recall** = "Of all actual positives, how many did I catch?" — important when FN is expensive (cancer screening).
- **F1-score** = harmonic mean of the two; balances both.

You can shift the threshold (default 0.5) to favor one over the other.

### 2.4 ROC Curve and AUC
The **ROC curve** plots TPR (recall) vs FPR at every possible threshold. **AUC** is the area under that curve.

- AUC = 1.0 → perfect
- AUC = 0.5 → random guessing
- AUC = 0.0 → perfectly wrong (flip the predictions)

```python
from sklearn.metrics import roc_curve, roc_auc_score
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
```

### 2.5 Regression Metrics
| Metric | What it tells you | Sensitive to outliers? |
|---|---|---|
| **MAE** | Average absolute error, same units as target | Less |
| **MSE** | Average squared error | More (penalizes big errors) |
| **RMSE** | MSE in original units | More |
| **R²** | Fraction of variance explained (0–1) | N/A |
| **MAPE** | Mean absolute % error | N/A (good for comparing across scales) |

---

## 3. Cross-Validation

A single train/test split is noisy. **K-Fold Cross-Validation** trains K times on different splits and averages the result.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.3f}  (+/- {scores.std():.3f})")
```

- **K = 5 or 10** is standard
- For classification, use `StratifiedKFold` so each fold has the same class balance
- For time-series data, use `TimeSeriesSplit` (never shuffle — future must not leak into past)

---

## 4. Algorithm Deep Dive

### 4.1 Logistic Regression
Despite the name, it's a **classification** algorithm. It models the probability of a class using the logistic (sigmoid) function.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Binary classification
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Key ideas**:
- Output is a probability (0–1), thresholded at 0.5 by default
- Coefficients tell you how each feature shifts the log-odds
- Works surprisingly well as a baseline — try it before fancier models
- Multinomial variant handles multi-class problems directly

### 4.2 Decision Trees
A tree of if/else questions. At each node, pick the feature and threshold that best separates the classes (or reduces variance for regression).

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
```

**Pros**: highly interpretable, handles mixed data types, no scaling needed.
**Cons**: prone to overfitting. A single tree has high variance — small data changes → very different tree.

**Controlling overfitting**:
- `max_depth` — limit tree depth
- `min_samples_split` / `min_samples_leaf` — require enough samples per node
- `criterion` — 'gini' (default) or 'entropy'

### 4.3 K-Means Clustering
The workhorse of unsupervised learning. Groups data into K clusters by minimizing within-cluster variance.

```python
from sklearn.cluster import KMeans

# Finding the right K with the elbow method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Final model
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)
```

**Limitations to know**:
- Must choose K in advance
- Sensitive to initialization (hence `n_init`)
- Assumes spherical, similarly-sized clusters — fails on weird shapes
- Always scale features first

**Picking K**: elbow method (inertia plot) or silhouette score.

---

## 5. Real-World Project Walkthrough: Titanic Survival Prediction

A classic end-to-end pipeline. We'll predict who survived the Titanic disaster.

### 5.1 The Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load
df = pd.read_csv('titanic.csv')

# 2. Quick exploration
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df['Survived'].value_counts())
```

### 5.2 Feature Engineering
Raw columns aren't enough — create signals the model can use.

```python
# Family size: alone vs with family
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Extract title from name
df['Title'] = df['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don',
                                   'Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss').replace('Mme', 'Mrs')
```

### 5.3 Build a Pipeline
A `Pipeline` chains preprocessing and modeling — and crucially, prevents leakage by fitting transformers only on training data.

```python
numeric_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline: preprocess then classify
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# 3. Train / test split
X = df[numeric_features + categorical_features]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Cross-validate
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 5. Fit on full training set and evaluate on test
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 5.4 What This Project Teaches
- **Exploration first** — checking `isnull()`, dtypes, and class balance before modeling
- **Feature engineering** beats fancier models: a `Title` feature often improves accuracy more than swapping classifiers
- **Pipelines prevent leakage** — no more accidentally fitting a scaler on test data
- **CV is the honest score** — a single test number can be lucky or unlucky
- **Stratified split** keeps class balance consistent across train and test

---

## 6. Hands-On Exercises

Try these before Day 3. Each takes 30–60 minutes.

1. **Logistic Regression on Iris**
   Load `sklearn.datasets.load_iris`. Fit a logistic regression. Use `cross_val_score` with 5 folds. Print the confusion matrix and classification report.

2. **Decision Tree Visualization**
   On the Titanic data, fit a `DecisionTreeClassifier(max_depth=3)` and use `plot_tree` to draw it. Identify which feature the tree splits on first.

3. **K-Means on Synthetic Data**
   Generate a blob dataset with `make_blobs(n_samples=300, centers=4, random_state=42)`. Plot the data, run K-Means for K = 2..6, and find the elbow.

4. **Metric Comparison**
   Build a deliberately imbalanced binary dataset (`make_classification(weights=[0.95, 0.05])`). Show that accuracy is misleading — a "predict all negative" model scores 95% but recall is 0.

5. **Build Your Own Pipeline**
   Pick any classification dataset from `sklearn.datasets`. Build a `Pipeline` with imputation, scaling, one-hot encoding, and a logistic regression. Compare cross-validated scores with vs without scaling.

---

## 7. Common Pitfalls (Day 2 Edition)

1. **Scaling before splitting** → information leaks from test into train. Always split first.
2. **Picking K by test accuracy** → you've used the test set for model selection, so it's no longer unbiased. Use validation set or CV.
3. **Cross-validating a pipeline that includes target-dependent steps** → the pipeline handles this correctly; manual loops often don't.
4. **Reporting only accuracy on imbalanced data** → always report precision, recall, F1, and ROC-AUC.
5. **Skipping baseline models** → a simple logistic regression often beats a poorly-tuned neural net. Always start simple.
6. **Forgetting `random_state`** → results aren't reproducible. Set it everywhere.

---

## 8. Summary for Day 2

1. **Preprocessing**: impute missing values, encode categoricals, scale features — fit on train, apply to test
2. **Metrics**: accuracy can lie; use precision/recall/F1 for imbalanced classification, MAE/RMSE/R² for regression, ROC-AUC for ranking quality
3. **Cross-validation**: 5- or 10-fold gives a more honest estimate than a single split
4. **Algorithms covered**: logistic regression (linear baseline), decision trees (interpretable but overfit-prone), K-means (unsupervised clustering)
5. **Pipelines**: chain preprocessing and modeling; prevent leakage; enable clean cross-validation
6. **Workflow**: explore → engineer features → build pipeline → cross-validate → evaluate on held-out test

**Next Steps (Day 3)**:
- Random Forests and ensemble methods
- Support Vector Machines
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Regularization (L1 / L2)
- A second end-to-end project with model comparison

Good luck — see you on Day 3! 🚀
