# Machine Learning Day 1 Tutorial Notes

## What is Machine Learning?

Machine Learning is a subset of Artificial Intelligence where computers learn from data to make predictions or decisions without being explicitly programmed. The idea is that the computer can learn from experience (data) just like humans do.

**Key Principle**: Given enough examples, the computer can find patterns and generalize to make predictions on new, unseen data.

---

## Types of Machine Learning

### 1. Supervised Learning
In supervised learning, we provide the algorithm with examples of correct answers.

**Characteristics**:
- Input data + corresponding output labels
- Algorithm learns the mapping from input to output
- Can make predictions on new data

**Types**:
- **Classification**: Predicts categories (e.g., spam or not spam)
  - Binary classification: Two classes (Yes/No, Spam/Not spam)
  - Multi-class classification: Multiple classes (Cat/Dog/Bird)
  - Example: Email filtering, sentiment analysis

- **Regression**: Predicts numerical values (e.g., house prices, temperature)
  - Linear regression: Straight line relationship
  - Polynomial regression: Curved relationships
  - Example: Stock price prediction, sales forecasting

**Popular Algorithms**:
- Linear Regression, Logistic Regression
- Decision Trees, Random Forests
- Support Vector Machines
- k-Nearest Neighbors

### 2. Unsupervised Learning
In unsupervised learning, we only provide input data without corresponding labels.

**Characteristics**:
- Algorithm finds patterns, structures, or groupings in data
- Discovers hidden relationships
- No predefined answers to guide learning

**Types**:
- **Clustering**: Groups similar data points together
  - K-means clustering: Divides data into k groups
  - Hierarchical clustering: Builds tree-like structure
  - Example: Customer segmentation, anomaly detection

- **Dimensionality Reduction**: Reduces number of features while preserving information
  - Principal Component Analysis (PCA): Finds most important features
  - t-SNE: Visualizes high-dimensional data
  - Example: Face recognition, data visualization

- **Association Rule Learning**: Finds relationships between variables
  - Apriori algorithm: Market basket analysis
  - Example: Product recommendation systems

### 3. Reinforcement Learning
This is less common for beginners but worth knowing.

**Characteristics**:
- Agent learns by interacting with an environment
- Receives rewards for good actions, penalties for bad actions
- Goal is to maximize cumulative reward

**Real-world Example**:
- Self-driving cars: Learn through trial and error
- Game playing: AlphaGo learned to beat world champions

---

## Key Concepts in ML

### 1. Features and Target Variables
- **Features (X)**: Input variables used for prediction
- **Target (y)**: What we want to predict

Example: House price prediction
- Features: Area, number of rooms, location
- Target: House price

### 2. Training and Testing
- **Training Data**: Used to teach the model
- **Testing Data**: Used to evaluate model performance

Important: Never use the same data for both training and testing!

### 3. Overfitting vs Underfitting
- **Overfitting**: Model learns training data too well, fails on new data
  - Complex model, perfect training performance, poor testing performance
  - Like memorizing answers for a test instead of understanding concepts

- **Underfitting**: Model is too simple to capture patterns
  - Simple model, poor performance on both training and testing
  - Like not studying enough for a test

**Goal**: Find the right balance (the "sweet spot")

### 4. Model Evaluation Metrics

**Classification Metrics**:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
  - Overall correctness
- **Precision**: TP / (TP + FP)
  - When we predict positive, how often are we correct?
- **Recall**: TP / (TP + FN)
  - Of all actual positives, how many did we find?
- **F1-Score**: Harmonic mean of precision and recall

**Regression Metrics**:
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Mean Squared Error (MSE)**: Average squared difference
- **R-squared**: Proportion of variance explained

---

## Introduction to Python for ML

### 1. Essential Libraries
```python
import numpy as np          # Numerical operations
import pandas as pd         # Data manipulation
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns        # Statistical plotting
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.linear_model import LinearRegression  # Linear regression
```

### 2. First ML Example: Linear Regression

**Problem**: Predict house prices based on area

**Steps**:
1. **Load and prepare data**
2. **Split into training/testing**
3. **Train the model**
4. **Make predictions**
5. **Evaluate performance**

**Code Example**:
```python
# 1. Prepare data
areas = np.array([500, 700, 1000, 1200, 1500])  # Features (input)
prices = np.array([100, 150, 250, 300, 400])      # Target (output)

# 2. Split the data (simple version - we'll use all for demo)
X = areas.reshape(-1, 1)  # Reshape for scikit-learn
y = prices

# 3. Split into training and testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"MSE: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
```

### 3. Understanding Model Parameters
In linear regression: **y = mx + b**

**What each parameter means**:
- **m (slope)**: How much y changes when x changes by 1 unit
- **b (intercept)**: y-value when x = 0

In our example:
- Slope tells us how much price changes per square foot
- Intercept is the base price of a 0-area house (theoretical)

---

## Common Mistakes to Avoid

1. **Don't use all your data for training!** 
   Always keep a testing set to evaluate real-world performance.

2. **Don't ignore data preprocessing**
   - Remove missing values
   - Normalize/scale features when needed
   - Check for outliers

3. **Don't trust accuracy alone**
   Use appropriate metrics for your problem type.

4. **Don't ignore the assumptions**
   Many algorithms have assumptions about your data.

5. **Don't stop at good training performance**
   The real test is how well it performs on NEW, unseen data.

---

## Summary for Day 1

1. **What is ML**: Computers learning from data to make predictions
2. **Three main types**: Supervised, Unsupervised, Reinforcement
3. **Key concepts**: Features, targets, training/testing, overfitting/underfitting
4. **Python basics**: Essential libraries and a simple linear regression example
5. **Common pitfalls**: Data splitting, preprocessing, proper evaluation

**Next Steps (Day 2)**:
- Deeper dive into model evaluation
- Introduction to more complex algorithms
- Real-world projects and case studies
- Advanced preprocessing techniques

Good luck with your ML journey! 🎯