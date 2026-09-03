# Machine Learning Day 4 Tutorial Notes

Building on Days 1–3, today we shift focus to unsupervised learning and neural networks — two areas that open up very different kinds of problems: finding hidden structure in data without labels, and learning complex non-linear patterns.

---

## 1. Unsupervised Learning Deep Dive

Unsupervised learning finds structure in data without any labels. No one tells the algorithm what "正确答案" is — it discovers patterns on its own.

### 1.1 Clustering: Grouping Similar Data Points

**What it is**: Partition data into clusters where items within a cluster are more similar to each other than to items in other clusters.

**When to use**: Customer segmentation, anomaly detection, image compression, document grouping.

#### K-Means Clustering (Recap & Advanced)

K-Means from Day 2 but with refinements:

```python
from sklearn.cluster import KMeans
import numpy as np

# K-Means with multiple initializations for stability
kmeans = KMeans(n_clusters=4, n_init=50, random_state=42, max_iter=500)
labels = kmeans.fit_predict(X_scaled)

# Inertia = within-cluster sum of squares (lower = tighter clusters)
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Number of iterations: {kmeans.n_iter_}")
```

**Choosing K — Beyond the Elbow**:

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

# Silhouette score: how similar an item is to its own cluster vs others
# Range: -1 to 1. Higher = better defined clusters
silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot silhouette analysis per sample
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
for idx, k in enumerate([2, 3, 4, 5]):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    sample_scores = silhouette_samples(X_scaled, labels)
    avg = silhouette_score(X_scaled, labels)

    y_lower = 10
    for i in range(k):
        ith_scores = sample_scores[labels == i]
        ith_scores.sort()
        size = len(ith_scores)
        y_upper = y_lower + size
        ax[idx // 2, idx % 2].fill_betweenx(
            np.arange(y_lower, y_upper), 0, ith_scores
        )
        y_lower = y_upper + 10
    ax[idx // 2, idx % 2].axvline(x=avg, color="red", linestyle="--")
    ax[idx // 2, idx % 2].set_title(f"K={k}, Avg Silhouette={avg:.3f}")
    ax[idx // 2, idx % 2].set_xlabel("Silhouette Coefficient")
```

#### Hierarchical (Agglomerative) Clustering

Builds a tree of clusters (dendrogram) without needing to pre-specify K.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create dendrogram to visualize hierarchy
linkage_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("Sample Index / Cluster Size")
plt.ylabel("Distance")
plt.show()

# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)
```

**Linkage methods**:
| Method | How it merges clusters | Best for |
|--------|----------------------|----------|
| `ward` | Minimizes variance increase | Most cases, default |
| `complete` | Max pairwise distance | Compact, equally-sized clusters |
| `average` | Mean pairwise distance | Robust to noise |
| `single` | Min pairwise distance | Non-elliptical shapes (risky) |

#### DBSCAN (Density-Based Clustering)

No need to specify K. Finds arbitrarily-shaped clusters and labels outliers.

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Find optimal epsilon using k-distance graph
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 4])  # 4th nearest neighbor distance

plt.plot(distances)
plt.title("K-Distance Graph (find the elbow)")
plt.xlabel("Points (sorted)")
plt.ylabel("4th Nearest Neighbor Distance")
plt.show()

# DBSCAN with tuned eps
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

print(f"Number of clusters: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}")
print(f"Noise points: {list(db_labels).count(-1)}")
```

**DBSCAN advantages**:
- Finds arbitrarily-shaped clusters (not just spherical)
- Automatically detects outliers (label = -1)
- No need to specify K
- Works well with noise

**DBSCAN limitations**:
- Sensitive to `eps` and `min_samples`
- Struggles with clusters of varying density

**Choosing a clustering algorithm**:

| Algorithm | K needed? | Arbitrary shapes? | Outliers? | Scalable? |
|---|---|---|---|---|
| K-Means | Yes | No (spherical) | No | Yes |
| Hierarchical | No (or after) | No | No | Medium |
| DBSCAN | No | Yes | Yes | Yes |

### 1.2 Dimensionality Reduction: Seeing High-Dimensional Data

Too many features → slow models, overfitting, impossible to visualize. Dimensionality reduction compresses the information into fewer dimensions.

#### Principal Component Analysis (PCA)

Finds orthogonal axes (principal components) that capture the most variance.

```python
from sklearn.decomposition import PCA

# PCA: find how many components explain 95% of variance
pca = PCA()
pca.fit(X_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")

# Plot explained variance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_, alpha=0.7)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance")
plt.tight_layout()
plt.show()

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)
print(f"Explained variance: {pca_2d.explained_variance_ratio_}")
```

**When to use PCA**:
- Speed up training (fewer features)
- Reduce overfitting
- Visualize high-dimensional data (2-3D)
- Remove multicollinearity before linear models
- Preprocessing before clustering

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction. Great for visualization, not for preprocessing.

```python
from sklearn.manifold import TSNE

# t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("t-SNE Visualization")
plt.show()
```

**PCA vs t-SNE**:

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| Linear? | Yes | No |
| Preserves structure | Global | Local |
| Speed | Fast | Slow |
| Good for | Preprocessing, ML features | Visualization |
| Reproducible? | Yes | No (random init) |

---

## 2. Introduction to Neural Networks

Neural networks learn non-linear relationships by stacking layers of simple neurons.

### 2.1 The Perceptron

A single neuron that makes a binary decision:

```python
import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, n_epochs=100):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                linear = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear >= 0 else 0
                update = self.lr * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return np.where(linear >= 0, 1, 0)
```

**Limitation**: A perceptron can only learn linearly separable problems (e.g., AND, OR — but NOT XOR).

### 2.2 Multi-Layer Perceptron (MLP)

Stack multiple layers of neurons to learn non-linear patterns (like XOR).

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 neurons, 50 neurons
    activation='relu',             # ReLU: max(0, x)
    solver='adam',                  # Adam optimizer (default, best for most cases)
    alpha=0.001,                   # L2 regularization
    batch_size='auto',
    learning_rate='adaptive',       # Reduce LR when progress stalls
    max_iter=500,
    random_state=42,
    early_stopping=True,            # Stop when validation score stops improving
    validation_fraction=0.1
)

mlp.fit(X_train_scaled, y_train)
y_pred = mlp.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Inspect the network
print(f"Layers: {mlp.hidden_layer_sizes}")
print(f"Loss curve: {mlpp.loss_curve_[-5:]}")  # Last 5 loss values
```

### 2.3 Activation Functions

| Function | Formula | Range | When to use |
|----------|---------|-------|-------------|
| **ReLU** | `max(0, x)` | [0, ∞) | Default for hidden layers |
| **Sigmoid** | `1 / (1 + exp(-x))` | (0, 1) | Binary output layer |
| **Tanh** | `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` | (-1, 1) | Hidden layers (zero-centered) |
| **Softmax** | `exp(x) / sum(exp(x))` | (0, 1) sum=1 | Multi-class output layer |

```python
# Visualize activation functions
x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(x, np.maximum(0, x)); axes[0, 0].set_title("ReLU")
axes[0, 1].plot(x, 1 / (1 + np.exp(-x))); axes[0, 1].set_title("Sigmoid")
axes[1, 0].plot(x, np.tanh(x)); axes[1, 0].set_title("Tanh")
axes[1, 1].plot(x, np.exp(x) / np.exp(x).sum()); axes[1, 1].set_title("Softmax (simulated)")
plt.tight_layout()
plt.show()
```

### 2.4 Neural Network Hyperparameters

| Hyperparameter | What it controls | Typical values |
|---|---|---|
| `hidden_layer_sizes` | Number and size of hidden layers | (100,), (64, 32), (128, 64, 32) |
| `activation` | Non-linearity | 'relu' (default) |
| `alpha` | L2 regularization strength | 0.0001–0.01 |
| `learning_rate_init` | Step size of weight updates | 0.001 (Adam), 0.01 (SGD) |
| `batch_size` | Samples per gradient update | 32–256, 'auto' |
| `max_iter` | Maximum training epochs | 200–1000 |
| `early_stopping` | Stop when val score plateaus | True |
| `validation_fraction` | Fraction for early stopping | 0.1 |

### 2.5 Learning Curves for Neural Networks

```python
# Monitor training vs validation loss to detect overfitting
from sklearn.model_selection import cross_val_predict

train_losses = []
val_scores = []

for alpha in [0.0001, 0.001, 0.01, 0.1]:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                       alpha=alpha, max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    train_losses.append(mlp.loss_curve_[-1])
    val_scores.append(mlp.score(X_val_scaled, y_val))

plt.plot([0.0001, 0.001, 0.01, 0.1], train_losses, 'o-', label='Training Loss')
plt.plot([0.0001, 0.001, 0.01, 0.1], val_scores, 's-', label='Validation Score')
plt.xscale('log')
plt.xlabel('Alpha (regularization strength)')
plt.legend()
plt.title('Learning Curve: Effect of Regularization')
plt.show()
```

---

## 3. Anomaly Detection

Finding rare events that differ significantly from the norm. Useful for fraud detection, network intrusion, equipment failure.

### 3.1 Simple Statistical Methods

```python
from scipy import stats

# Z-score method: flag points beyond 3 standard deviations
z_scores = np.abs(stats.zscore(X_numeric))
outliers = (z_scores > 3).any(axis=1)
print(f"Outliers detected: {outliers.sum()}")

# IQR method
Q1, Q3 = X_numeric.quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers_iqr = ((X_numeric < (Q1 - 1.5 * IQR)) | (X_numeric > (Q3 + 1.5 * IQR))).any(axis=1)
```

### 3.2 Isolation Forest

Tree-based anomaly detection. Anomalies are easier to isolate (fewer splits needed).

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # Expected fraction of outliers
    random_state=42
)
outlier_labels = iso_forest.fit_predict(X_scaled)

# -1 = outlier, 1 = inlier
print(f"Outliers: {(outlier_labels == -1).sum()}")
```

### 3.3 Local Outlier Factor (LOF)

Compares local density of a point to its neighbors. Points with much lower density than neighbors are outliers.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outlier_labels = lof.fit_predict(X_scaled)
print(f"Outliers: {(outlier_labels == -1).sum()}")
```

---

## 4. Model Deployment & Production Considerations

Getting a model to production is different from notebooks. Here's what matters.

### 4.1 Saving and Loading Models

```python
import joblib

# Save a trained model
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Load in production
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Predictions
X_new = scaler.transform(new_data)
predictions = model.predict(X_new)
```

### 4.2 Production Pipeline Checklist

```
1. Save the full preprocessing + model pipeline, not just the model
2. Document the expected input format and output format
3. Log predictions for monitoring and retraining
4. Set up model monitoring: track accuracy drift, data drift
5. Plan for retraining: when to retrain? based on performance metrics?
6. Version your models: model_v1, model_v2, etc.
```

### 4.3 Train Once, Predict Forever

```python
# Save everything needed for prediction
import json
import joblib

pipeline = {
    'model': trained_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'model_version': '1.0.0',
    'training_date': '2024-09-04',
    'performance': {
        'cv_accuracy': 0.87,
        'test_accuracy': 0.85
    }
}

joblib.dump(pipeline, 'production_pipeline.joblib')

# Load and predict
loaded = joblib.load('production_pipeline.joblib')
X_new_scaled = loaded['scaler'].transform(new_data)
preds = loaded['model'].predict(X_new_scaled)
```

### 4.4 Model Monitoring Basics

Track these over time:
- **Prediction distribution** — has it shifted?
- **Feature distribution** — is input data changing?
- **Accuracy** (if labels come back) — is the model degrading?
- **Latency** — are predictions getting slower?

---

## 5. End-to-End Project: Mall Customers Segmentation

Use clustering to segment mall customers, then use PCA/t-SNE to understand the segments.

### 5.1 Data Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Load data (assuming CSV with these columns)
df = pd.read_csv('mall_customers.csv')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for idx, col in enumerate(numerical_cols):
    ax = axes[idx // 2, idx % 2]
    ax.hist(df[col], bins=20, edgecolor='black', alpha=0.7)
    ax.set_title(f"{col} Distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

# Scatter plot: Annual Income vs Spending Score
axes[1, 1].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                    alpha=0.6, c='steelblue')
axes[1, 1].set_xlabel("Annual Income (k$)")
axes[1, 1].set_ylabel("Spending Score (1-100)")
axes[1, 1].set_title("Income vs Spending Score")

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr(),
            annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()
```

### 5.2 Feature Selection and Scaling

```python
# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Features: {features}")
print(f"X_scaled shape: {X_scaled.shape}")
```

### 5.3 Finding the Optimal Number of Clusters

```python
# K-Means: Elbow + Silhouette method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-cluster Sum of Squares)")
axes[0].set_title("Elbow Method")
axes[0].grid(True, alpha=0.3)

# Silhouette plot
axes[1].plot(K_range, silhouette_scores, 'go-')
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
print(f"Best K by Silhouette Score: {best_k} (score: {max(silhouette_scores):.3f})")
```

### 5.4 Compare Multiple Clustering Algorithms

```python
# Apply multiple clustering methods
results = {}

# K-Means
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
results['K-Means'] = {
    'labels': kmeans_labels,
    'silhouette': silhouette_score(X_scaled, kmeans_labels)
}

# Hierarchical (Ward)
hier = AgglomerativeClustering(n_clusters=5, linkage='ward')
hier_labels = hier.fit_predict(X_scaled)
results['Hierarchical'] = {
    'labels': hier_labels,
    'silhouette': silhouette_score(X_scaled, hier_labels)
}

# DBSCAN (tune eps using k-distance graph first)
dbscan = DBSCAN(eps=0.7, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)
results['DBSCAN'] = {
    'labels': db_labels,
    'silhouette': silhouette_score(X_scaled, db_labels) if len(set(db_labels)) > 1 else None
}

# Print comparison
for name, res in results.items():
    print(f"{name}: Silhouette = {res['silhouette']:.3f}" if res['silhouette'] else f"{name}: DBSCAN noise points = {list(db_labels).count(-1)}")
```

### 5.5 Visualize the Best Clustering

```python
# Use PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE for comparison
tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original data (no clustering)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
axes[0].set_title("PCA - No Clustering")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# K-Means clusters
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=200, edgecolors='black')
axes[1].set_title(f"K-Means (K=5) — Silhouette: {results['K-Means']['silhouette']:.3f}")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

# t-SNE visualization
axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
axes[2].set_title("t-SNE with K-Means Clusters")
axes[2].set_xlabel("t-SNE 1")
axes[2].set_ylabel("t-SNE 2")

plt.tight_layout()
plt.show()
```

### 5.6 Analyze the Customer Segments

```python
# Add cluster labels to dataframe
df['Segment'] = kmeans_labels

# Analyze each segment
print("Customer Segment Analysis:\n")
segment_summary = df.groupby('Segment')[features].mean()
print(segment_summary.round(2))

# Visualize segment profiles
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(features):
    df.boxplot(column=feature, by='Segment', ax=axes[idx])
    axes[idx].set_title(f"{feature} by Segment")
    axes[idx].set_xlabel("Segment")
plt.suptitle("Customer Segment Profiles", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Naming the segments (based on the data patterns)
segment_names = {
    0: "Budget Spenders",      # Low income, high spending
    1: "Careful Savers",       # Low income, low spending
    2: "Average Customers",   # Middle income, middle spending
    3: "High Earners",         # High income, low spending
    4: "VIP Customers"         # High income, high spending
}

df['Segment Name'] = df['Segment'].map(segment_names)
print("\nSegment Distribution:")
print(df['Segment Name'].value_counts())
```

### 5.7 Key Takeaways from the Project

- **Always scale before clustering** — K-Means and DBSCAN are distance-based
- **Multiple algorithms** — K-Means, Hierarchical, and DBSCAN can give very different results
- **Silhouette + elbow together** — neither alone tells the full story
- **PCA for ML features** — use PCA-reduced dimensions as inputs to supervised models
- **t-SNE for visualization** — don't feed t-SNE output to a downstream model (it's non-linear and stochastic)
- **Business interpretation** — clustering is only useful if the segments are actionable

---

## 6. Hands-On Exercises

Try these before Day 5. Each takes 45–90 minutes.

1. **Find the Perfect Cluster**
   Generate synthetic data with `make_blobs` (3 clusters, 2 features). Add some noise points. Run K-Means, Hierarchical, and DBSCAN. Which algorithm handles the noise best? Try varying `contamination` in DBSCAN.

2. **PCA Image Compression**
   Load any image as a numpy array. Apply PCA to compress it to different numbers of components (50, 100, 200). Visualize the original vs compressed images. At what number of components does the image look visually similar?

3. **Neural Network on MNIST (or a subset)**
   Use sklearn's MLPClassifier on a subset of the digits dataset. Plot the learning curve. Try different architectures: (50,), (100, 50), (100, 50, 25). Which converges fastest?

4. **Anomaly Detection Comparison**
   Generate data with 5% outliers. Compare Z-score, Isolation Forest, and LOF on the same data. Visualize which points each method flags as anomalies. Do they agree?

5. **Customer Segmentation Refinement**
   Extend the mall customers project: add more features (Gender, Profession). How does gender distribution vary across segments? Build a simple supervised model (logistic regression or random forest) to predict segment membership.

6. **t-SNE vs PCA for Visualization**
   On the Iris dataset, apply PCA and t-SNE to reduce to 2D. Compare the cluster separation. Run K-Means on both reduced datasets and compare silhouette scores. Which representation is better for clustering?

---

## 7. Common Pitfalls (Day 4 Edition)

1. **Clustering without scaling** — distance-based algorithms (K-Means, DBSCAN, Hierarchical) are meaningless on unscaled features
2. **Using t-SNE for downstream tasks** — t-SNE is for visualization only, not feature engineering
3. **Picking K from elbow alone** — silhouette analysis gives a second opinion
4. **Over-interpreting random t-SNE output** — run t-SNE multiple times; different runs produce different layouts
5. **Neural network overfitting** — start with a small architecture and use early stopping + regularization
6. **Forgetting to save the scaler** — a model without its preprocessing pipeline is useless in production
7. **Deploying models without monitoring** — data drift silently degrades performance

---

## 8. Summary for Day 4

1. **Clustering algorithms** — K-Means (needs K, spherical clusters), Hierarchical (dendrogram), DBSCAN (density-based, outlier detection)
2. **Dimensionality reduction** — PCA (linear, for ML preprocessing), t-SNE (non-linear, for visualization only)
3. **Neural networks** — Perceptron, MLP with hidden layers, activation functions, key hyperparameters
4. **Anomaly detection** — Z-score, Isolation Forest, Local Outlier Factor
5. **Production basics** — saving/loading models with joblib, pipeline reproducibility, monitoring
6. **End-to-end project** — Mall Customers Segmentation with multiple clustering algorithms and dimensionality reduction

**Next Steps (Day 5)**:
- Introduction to Deep Learning frameworks (TensorFlow / PyTorch basics)
- Convolutional Neural Networks (CNNs) for image data
- Recurrent Neural Networks (RNNs) and sequence data
- Natural Language Processing (NLP) basics
- Transfer learning and pre-trained models
- Final project: Build and deploy a simple ML model
- Course wrap-up and next steps

Ready for Day 5! 🚀
