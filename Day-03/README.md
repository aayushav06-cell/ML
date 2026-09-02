# Machine Learning Day 3 Tutorial Notes

Building on the foundations from Days 1 and 2, today we dive deeper into more advanced algorithms and techniques. We'll explore ensemble methods, Support Vector Machines, and hyperparameter tuning to make our models even more powerful.

---

## 1. Ensemble Methods

Single models often have high variance or bias. **Ensemble methods** combine multiple models to improve performance, reduce overfitting, and increase stability.

### 1.1 Random Forest
Random Forest is an ensemble of decision trees. Each tree is trained on a random sample of data (bootstrap sampling) and at each node, only a random subset of features is considered.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Create a RandomForest
rf = RandomForestClassifier(
    n_estimators=100,          # Number of trees
    max_depth=10,              # Limit depth to reduce overfitting
    min_samples_split=5,       # Require at least 5 samples to split
    random_state=42
)

# Train and evaluate
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

**Hyperparameters to tune**:
- `n_estimators` — more trees → better but slower
- `max_depth` — limit depth to prevent overfitting
- `min_samples_split` / `min_samples_leaf` — control tree growth
- `max_features` — number of features considered at each split
- `bootstrap` — whether to use bootstrap samples

**Feature importance**:
```python
import matplotlib.pyplot as plt
import numpy as np

feature_importance = rf.feature_importances_
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

### 1.2 Gradient Boosting & AdaBoost

**Gradient Boosting** builds models sequentially, with each new model correcting the errors of the previous ones. It's powerful but can overfit easily.

**AdaBoost** (Adaptive Boosting) is simpler — it weights misclassified instances more heavily in subsequent models.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    random_state=42
)
```

### 1.3 Stacking & Blending
Ensemble techniques that combine different types of models:

- **Stacking** — train a meta-model to combine predictions
- **Blending** — average predictions from models trained on different data subsets

```python
# Stacking example
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)
```

---

## 2. Support Vector Machines (SVM)

SVM is a powerful classification algorithm that finds the optimal hyperplane that maximizes the margin between classes.

### 2.1 What Makes SVM Special

- **Maximum margin** — the hyperplane furthest from both classes
- **Kernel trick** — transforms data to higher dimensions where it's linearly separable
- **Support vectors** — data points that define the decision boundary

### 2.2 Kernels

| Kernel | When to use | Best for |
|--------|-------------|----------|
| **Linear** | High-dimensional data, many features | Text classification, linear relationships |
| **Polynomial** | Complex boundaries, curved decision surfaces | Moderate non-linearity |
| **RBF (Radial Basis Function)** | Complex patterns, default choice | Most general-purpose problems |
| **Sigmoid** | Neural network-like behavior | Specific similarity functions |

```python
from sklearn.svm import SVC

# Linear kernel - fast, works well with many features
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)

# RBF kernel - most flexible, slower but powerful
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
```

### 2.3 Hyperparameters

- **C (Regularization)** — controls tradeoff between margin width and misclassification:
  - Small C → wide margin, more misclassifications
  - Large C → narrow margin, fewer misclassifications

- **Gamma** (for RBF/Poly) — how far the influence of a single example reaches:
  - Small gamma → far reach, smoother boundary
  - Large gamma → local influence, wiggly boundary

- **Nu** (for NuSVC) — upper bound on margin errors and lower bound on support vectors

### 2.4 One-vs-Rest & Multi-Class

For multi-class problems, SVM uses strategies like:

- **One-vs-Rest (OvR)** — train one classifier per class
- **One-vs-One (OvO)** — train classifiers for every pair of classes

```python
# Multi-class SVM using OvR
svm_multi = SVC(decision_function_shape='ovr', random_state=42)

# Multi-class SVM using OvO  
svm_multi_ovo = SVC(decision_function_shape='ovo', random_state=42)
```

---

## 3. Hyperparameter Tuning

The best model comes from finding the right hyperparameters. Automated tuning saves time and finds better combinations than manual trial-and-error.

### 3.1 Grid Search
**GridSearchCV** exhaustively searches a predefined parameter grid using cross-validation.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Create GridSearchCV
lr = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Predictions with best model
y_pred = grid_search.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### 3.2 Randomized Search
**RandomizedSearchCV** samples parameters from distributions instead of exhaustive search. Faster and often finds good solutions.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import numpy as np

# Define distributions
param_dist = {
    'n_estimators': stats.randint(50, 300),
    'max_depth': stats.randint(3, 20),
    'min_samples_split': stats.randint(2, 11),
    'min_samples_leaf': stats.randint(1, 6),
    'max_features': stats.uniform(0.5, 0.5)
}

# Create RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_dist, n_iter=50, cv=5, scoring='accuracy',
    random_state=42, n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

### 3.3 Validation Strategies

| Strategy | Best for | How it works |
|----------|----------|--------------|
| **K-Fold** | General use | Splits data into K folds, uses K-1 for training |
| **StratifiedKFold** | Imbalanced data | Ensures class balance in each fold |
| **TimeSeriesSplit** | Temporal data | Respects time order, no shuffling |
| **LeaveOneOut** | Very small datasets | Uses one sample as test, rest as train |

```python
from sklearn.model_selection import StratifiedKFold

# Stratified 5-fold CV
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# TimeSeriesSplit
scorer = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in scorer.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # train/test logic here
```

---

## 4. Regularization

Regularization adds penalty terms to the loss function to prevent overfitting by discouraging complex models.

### 4.1 L1 (Lasso) & L2 (Ridge) Regularization

- **L1 (Lasso)** — adds sum of absolute values, can shrink coefficients to zero (feature selection)
- **L2 (Ridge)** — adds sum of squared values, shrinks coefficients but rarely to zero

```python
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

# L2 Regularization (Ridge)
ridge = Ridge(alpha=1.0, random_state=42)

# L1 Regularization (Lasso)
lasso = Lasso(alpha=0.1, random_state=42)

# Logistic Regression with regularization
lr_l2 = LogisticRegression(penalty='l2', C=1.0, solver='saga', max_iter=1000)
lr_l1 = LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=1000)
```

**Choosing between L1/L2**:
- **L1** when you want feature selection (many coefficients become zero)
- **L2** when you want to keep all features but reduce their impact
- **Elastic Net** combines both approaches

### 4.2 Early Stopping

For iterative models (gradient boosting, neural networks), stop training when validation performance stops improving.

```python
# Using early stopping with cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# Monitor performance across iterations
gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Use cross-validation to find optimal n_estimators
scores = []
for n in [50, 100, 200, 500, 1000]:
    gb_tmp = GradientBoostingClassifier(
        n_estimators=n,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    score = np.mean(cross_val_score(gb_tmp, X_train, y_train, cv=5))
    scores.append(score)

best_n = [50, 100, 200, 500, 1000][np.argmax(scores)]
print(f"Best n_estimators: {best_n}")
```

---

## 5. Second End-to-End Project: Iris Flower Classification with Model Comparison

Let's build a complete pipeline that compares multiple algorithms on the classic Iris dataset.

### 5.1 The Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and explore the data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map(dict(enumerate(iris.target_names)))

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['target_name'].value_counts())

# 2. Split the data
X = df[iris.feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5.2 PCA for Visualization

Reduce to 2 dimensions for easy visualization.

```python
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                     c=y_train, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Iris Data - PCA (Training Set)')
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

### 5.3 Model Comparison

Train multiple algorithms and compare their performance.

```python
# Define all models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
    'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Cross-validation results
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=5, scoring='accuracy')
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': None
    }
    print(f"{name}: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Train final models and evaluate on test set
final_results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_score = accuracy_score(y_test, y_pred)
    final_results[name] = {
        'model': model,
        'test_score': test_score,
        'predictions': y_pred
    }
    print(f"{name} test accuracy: {test_score:.3f}")
```

### 5.4 Performance Comparison

Visualize the results side by side.

```python
# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Plot 1: CV Accuracy Comparison
model_names = list(results.keys())
cv_means = [results[name]['cv_mean'] for name in model_names]
cv_stds = [results[name]['cv_std'] for name in model_names]

axes[0].barh(model_names, cv_means, xerr=cv_stds, capsize=5, alpha=0.7)
axes[0].set_xlabel('Cross-Validation Accuracy')
axes[0].set_title('Model Comparison (CV)')
axes[0].axvline(x=np.mean(cv_means), color='red', linestyle='--', alpha=0.5)

# Plot 2: Test Accuracy Comparison
test_scores = [final_results[name]['test_score'] for name in model_names]

axes[1].barh(model_names, test_scores, capsize=5, alpha=0.7, color='lightgreen')
axes[1].set_xlabel('Test Accuracy')
axes[1].set_title('Model Comparison (Test Set)')
axes[1].axvline(x=np.mean(test_scores), color='red', linestyle='--', alpha=0.5)

# Plot 3: Confusion Matrix for best CV model
best_cv_name = model_names[np.argmax(cv_means)]
best_model = final_results[best_cv_name]['model']
y_pred = final_results[best_cv_name]['predictions']
cm = confusion_matrix(y_test, y_pred)

im = axes[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[2].set_title(f'Confusion Matrix ({best_cv_name})')
plt.colorbar(im, ax=axes[2])

# Add text to confusion matrix cells
for i in range(len(cm)):
    for j in range(len(cm[i])):
        axes[2].text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

# Plot 4: Learning curves for best model
from sklearn.model_selection import learning_curve

model = final_results[best_cv_name]['model']
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

axes[3].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
axes[3].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Cross-validation score')
axes[3].set_xlabel('Training examples')
axes[3].set_ylabel('Accuracy')
axes[3].set_title('Learning Curve')
axes[3].legend(loc='best')
axes[3].grid(True, alpha=0.3)

# Plot 5: Feature importance (for tree-based models)
for idx, (name, model_obj) in enumerate([(n, m) for n, m in models.items() if 'Tree' in n or 'Forest' in n]):
    if hasattr(model_obj, 'feature_importances_'):
        feat_imp = model_obj.feature_importances_
        axes[4 + idx].barh(range(len(feat_imp)), feat_imp)
        axes[4 + idx].set_title(f"{name} - Feature Importance")
        axes[4 + idx].set_xlabel('Importance')
        axes[4 + idx].set_ylabel('Feature Index')

# Hide unused subplots
for idx in range(4 + len([n for n in models.keys() if 'Tree' in n or 'Forest' in n])):
    if idx < 6 and idx not in [0, 1, 2, 3] and idx >= 4 + len([n for n in models.keys() if 'Tree' in n or 'Forest' in n]):
        axes[idx].set_visible(False)

plt.tight_layout()
plt.show()
```

### 5.5 Key Takeaways

- **Performance varies** — sometimes simple models beat complex ones
- **Cross-validation** is crucial for reliable comparison
- **Overfitting is common** — tree models often need tuning
- **Feature scaling matters** — SVMs and distance-based algorithms especially
- **Learning curves** help diagnose bias vs variance issues

---

## 6. Hands-On Exercises

Try these before Day 4. Each takes 45–90 minutes.

1. **SVM Hyperparameter Tuning**
   On the Iris dataset, use GridSearchCV to optimize an SVM. Try different kernels (linear, rbf, poly) and their hyperparameters. Plot the validation curves.

2. **Random Forest vs Gradient Boosting**
   Train both algorithms on the Titanic dataset. Compare their performance and overfitting. Use learning curves to diagnose issues. Try tuning hyperparameters using RandomizedSearchCV.

3. **Regularization Effects**
   On a linear classification problem (e.g., letter recognition), compare L1 and L2 regularization. Which features get selected with L1? How does L2 affect coefficient magnitudes?

4. **Early Stopping Experiment**
   With GradientBoostingClassifier, experiment with early stopping. Plot training vs validation accuracy as n_estimators increases. Find the optimal stopping point.

5. **Stacking Ensemble**
   Create a stacking classifier with at least 3 base models (different algorithms). Train it on the breast cancer dataset and compare to individual models and simple ensemble methods (averaging).

6. **Real-world Model Comparison**
   Choose any Kaggle dataset (heart disease, diabetes, etc.). Build a complete pipeline with preprocessing, multiple models, and comprehensive evaluation. Include hyperparameter tuning for at least one model.

---

## 7. Common Pitfalls (Day 3 Edition)

1. **Overfitting with complex models** — tree-based models and SVMs with RBF kernel overfit easily
2. **Ignoring feature scaling** — SVMs, KNN, and distance-based algorithms are sensitive to scale
3. **Wrong validation strategy** — using KFold on time-series or imbalanced data
4. **Over-tuning** — using too many hyperparameters without enough data
5. **Not comparing to simple baselines** — logistic regression often beats fancier models
6. **Feature leakage** — scaling or imputing on entire dataset before splitting
7. **Ignoring class imbalance** — accuracy is misleading when classes are imbalanced

---

## 8. Summary for Day 3

1. **Ensemble methods** — Random Forest (robust), Gradient Boosting (powerful), AdaBoost (simple), Stacking (hybrid)
2. **Support Vector Machines** — kernels (linear, poly, rbf), regularization (C), multi-class strategies
3. **Hyperparameter tuning** — GridSearchCV (exhaustive), RandomizedSearchCV (sampled), validation strategies
4. **Regularization** — L1 (feature selection), L2 (coefficient shrinkage), early stopping
5. **Complete pipeline** — preprocessing → model comparison → evaluation → visualization
6. **Real-world project** — fromEDA to model selection with comprehensive metrics

**Next Steps (Day 4)**:
- Neural networks and deep learning basics
- Clustering algorithms (K-means, hierarchical, DBSCAN)
- Unsupervised learning with PCA and t-SNE
- Model deployment and production considerations
- A second advanced project with real-world data

Ready for Day 4! 🎯
