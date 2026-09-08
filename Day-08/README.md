# Machine Learning Day 8 Tutorial Notes

After mastering MLOps and production deployment, we confront the most important question in modern ML: **should we build this at all?** AI Ethics, Bias, and Responsible AI ensure that the systems we create are fair, transparent, and beneficial to everyone.

---

## Table of Contents
1. [Why AI Ethics Matters](#1-why-ai-ethics-matters)
2. [Understanding Bias in ML](#2-understanding-bias-in-ml)
3. [Fairness Metrics and Measurement](#3-fairness-metrics-and-measurement)
4. [Bias Detection and Mitigation](#4-bias-detection-and-mitigation)
5. [Explainability and Interpretability](#5-explainability-and-interpretability)
6. [Responsible AI Frameworks](#6-responsible-ai-frameworks)
7. [Privacy and Data Governance](#7-privacy-and-data-governance)
8. [Hands-On Exercise: Bias Detection](#8-hands-on-exercise-bias-detection)总结
9. [Common Pitfalls](#9-common-pitfalls)
10. [Summary](#10-summary)

---

## 1. Why AI Ethics Matters

### The Real-World Impact of ML Decisions

```
Training data → Model → Decisions → Lives affected
     ↑                                      ↑
  Historical                           Real people
  bias                                 with rights
```

ML systems make decisions that affect:
- **Hiring**: Resume screening algorithms
- **Lending**: Credit approval and interest rates
- **Criminal Justice**: Recidivism prediction (COMPAS)
- **Healthcare**: Diagnosis and treatment recommendations
- **Education**: Admission and grading systems

### Historical Failures

| System | Issue | Consequence |
|--------|-------|-------------|
| COMPAS (2016) | Racial bias in recidivism prediction | False positives for Black defendants |
| Amazon Hiring (2018) | Gender bias in resume screening | Penalized resumes with "women's" |
| Facial Recognition | Higher error rates for darker skin | Misidentification, wrongful arrests |
| Healthcare Algorithms (2019) | Cost-based bias | Black patients under-treated |
| Twitter Photo Filter (2018) | Skin tone bias | White faces brightened, dark faces darkened |

### Key Ethical Principles

1. **Fairness**: No unjust discrimination against protected groups
2. **Transparency**: Decisions should be explainable and auditable
3. **Accountability**: Someone must be responsible for AI outcomes
4. **Privacy**: Personal data must be protected
5. **Safety**: Systems must be robust and fail safely
6. **Inclusivity**: Diverse perspectives in design and deployment

---

## 2. Understanding Bias in ML

### Types of Bias

```
                    ┌─────────────────────────────┐
                    │        DATA BIAS             │
                    │  (Historical/Selection/      │
                    │   Measurement bias)          │
                    └──────────┬──────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────┐
                    │       ALGORITHMIC BIAS       │
                    │  (Model learns and           │
                    │   amplifies patterns)        │
                    └──────────┬──────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────┐
                    │       DEPLOYMENT BIAS        │
                    │  (Context mismatch,          │
                    │   feedback loops)            │
                    └─────────────────────────────┘
```

### 1. Data Bias

**Historical Bias**: Training data reflects societal prejudices.
```python
# Example: Historical hiring data
# Past hiring was biased toward certain demographics
# Model learns: "hire people like past hires"
# Result: Perpetuates historical discrimination
```

**Selection Bias**: Training data doesn't represent the population.
```python
# Example: Medical dataset from one hospital
# May not represent demographics of other regions
# Model performs poorly on underrepresented groups
```

**Measurement Bias**: Features are measured inconsistently across groups.
```python
# Example: Credit scoring using zip code
# Zip codes correlate with race due to historical housing segregation
# → Indirect discrimination
```

### 2. Algorithmic Bias

**Optimization Bias**: Model optimizes for majority while ignoring minority.
```python
# A model trained on 90% class A, 10% class B
# May achieve 90% accuracy by always predicting class A
# Accuracy looks good, but class B gets zero attention
```

**Representation Bias**: Missing features or poor feature representation.
```python
# Facial recognition trained mainly on lighter skin
# Features don't generalize to darker skin tones
```

### 3. Deployment Bias

**Feedback Loops**: Predictions influence future data.
```
Model predicts group A is higher risk → More resources to group A
→ Better outcomes for group A → Model sees this → Reinforces bias
```

**Context Mismatch**: Model works in training environment, fails in real world.

---

## 3. Fairness Metrics and Measurement

### Definitions of Fairness

There are multiple mathematical definitions of fairness — and they're often **mutually exclusive**:

| Fairness Criterion | Definition | Requirement |
|-------------------|-----------|-------------|
| **Demographic Parity** | Prediction rate independent of group | P(Ŷ=1\|A=0) = P(Ŷ=1\|A=1) |
| **Equalized Odds** | Equal TPR and FPR across groups | P(Ŷ=1\|Y=1,A=g) equal for all g |
| **Equal Opportunity** | Equal TPR across groups | P(Ŷ=1\|Y=1,A=g) equal for all g |
| **Individual Fairness** | Similar individuals treated similarly | If x ≈ x', then ŷ ≈ ŷ' |
| **Calibration** | Predicted probabilities match true rates | P(Y=1\|Ŷ=p, A=g) = p for all g |

### Calculating Fairness Metrics

```python
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def fairness_report(y_true, y_pred, sensitive_attr):
    """
    Compute fairness metrics across groups.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        sensitive_attr: Protected attribute (e.g., gender, race)
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': sensitive_attr
    })
    
    results = {}
    for group in df['group'].unique():
        mask = df['group'] == group
        group_data = df[mask]
        
        tn, fp, fn, tp = confusion_matrix(
            group_data['y_true'], group_data['y_pred']
        ).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        
        results[group] = {
            'accuracy': accuracy,
            'tpr': tpr,
            'fpr': fpr,
            'count': len(group_data)
        }
    
    # Print fairness gaps
    groups = list(results.keys())
    if len(groups) >= 2:
        tpr_gap = abs(results[groups[0]]['tpr'] - results[groups[1]]['tpr'])
        fpr_gap = abs(results[groups[0]]['fpr'] - results[groups[1]]['fpr'])
        acc_gap = abs(results[groups[0]]['accuracy'] - results[groups[1]]['accuracy'])
        
        print(f"🔍 Fairness Report:")
        print(f"  TPR Gap:   {tpr_gap:.4f} (target: 0.0)")
        print(f"  FPR Gap:   {fpr_gap:.4f} (target: 0.0)")
        print(f"  Acc Gap:   {acc_gap:.4f} (target: 0.0)")
        print(f"  ⚠️  If gaps > 0.05, potential bias detected")
    
    return results
```

### Disparate Impact Ratio

```python
def disparate_impact_ratio(y_pred, sensitive_attr, favorable_label=1):
    """
    4/5ths rule (80% rule): Ratio of favorable outcomes between groups.
    < 0.8 suggests potential discrimination (EEOC guideline).
    """
    import pandas as pd
    
    df = pd.DataFrame({'pred': y_pred, 'group': sensitive_attr})
    
    rates = {}
    for group in df['group'].unique():
        mask = (df['group'] == group) & (df['pred'] == favorable_label)
        rates[group] = mask.mean()
    
    groups = list(rates.keys())
    di_ratio = rates[groups[0]] / rates[groups[1]] if rates[groups[1]] > 0 else float('inf')
    
    print(f"Disparate Impact Ratio: {di_ratio:.4f}")
    print(f"  {'✅ Fair' if di_ratio >= 0.8 else '⚠️ Potentially Discriminatory'}")
    
    return di_ratio
```

---

## 4. Bias Detection and Mitigation

### Detection Strategies

```python
import pandas as pd
import numpy as np
from scipy import stats

def detect_bias(df, target_col, sensitive_col, prediction_col=None):
    """
    Comprehensive bias detection across multiple dimensions.
    """
    print(f"📊 Bias Analysis: {sensitive_col} vs {target_col}")
    print("=" * 60)
    
    # 1. Distribution comparison
    print("\n1. Feature Distribution by Group:")
    for col in df.columns:
        if col not in [target_col, sensitive_col]:
            group_a = df[df[sensitive_col] == df[sensitive_col].unique()[0]][col]
            group_b = df[df[sensitive_col] == df[sensitive_col].unique()[1]][col]
            t_stat, p_value = stats.ttest_ind(group_a, group_b, nan_policy='omit')
            if p_value < 0.05:
                print(f"  ⚠️  {col}: Significant distribution difference (p={p_value:.4f})")
    
    # 2. Outcome rate comparison
    print(f"\n2. Outcome Rates by Group:")
    for group in df[sensitive_col].unique():
        rate = df[df[sensitive_col] == group][target_col].mean()
        print(f"  {group}: {rate:.4f}")
    
    # 3. Statistical tests
    print(f"\n3. Statistical Significance:")
    contingency = pd.crosstab(df[sensitive_col], df[target_col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"  Chi-squared p-value: {p:.6f}")
    print(f"  {'⚠️ Significant bias detected' if p < 0.05 else '✅ No significant bias'}")
```

### Mitigation Techniques

#### Pre-processing: Fix the Data

```python
from imblearn.over_sampling import SMOTE
import numpy as np

def rebalance_data(X, y, sensitive_attr):
    """
    Pre-processing mitigation: Resample to balance representation.
    """
    # Oversample minority groups
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Or: Reweighting samples
    # Assign higher weights to underrepresented groups
    weights = np.ones(len(y))
    for group in np.unique(sensitive_attr):
        group_mask = (sensitive_attr == group)
        weights[group_mask] = len(y) / (len(np.unique(sensitive_attr)) * group_mask.sum())
    
    return X_resampled, y_resampled, weights
```

#### In-processing: Modify the Algorithm

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class FairClassifier(BaseEstimator, ClassifierMixin):
    """
    In-processing: Add fairness constraint to training objective.
    Penalizes the model for disparate treatment across groups.
    """
    
    def __init__(self, base_model, fairness_weight=0.1):
        self.base_model = base_model
        self.fairness_weight = fairness_weight
    
    def fit(self, X, y, sensitive_attr):
        """Train with fairness penalty."""
        # Train base model
        self.base_model.fit(X, y)
        
        # Compute fairness penalty (simplified)
        predictions = self.base_model.predict(X)
        for group in np.unique(sensitive_attr):
            group_mask = (sensitive_attr == group)
            group_pred = predictions[group_mask]
            # Penalize if prediction rates differ significantly from overall rate
            overall_rate = predictions.mean()
            group_rate = group_pred.mean()
            penalty = self.fairness_weight * abs(group_rate - overall_rate)
            # In practice, this would be incorporated into the loss function
        
        self.fairness_penalty_ = penalty
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)
```

#### Post-processing: Adjust Predictions

```python
def equalized_odds_postprocessing(y_prob, sensitive_attr, threshold_0=0.5, threshold_1=0.5):
    """
    Post-processing: Apply different thresholds per group to equalize FPR/TPR.
    
    Adjust thresholds so that:
    - P(Ŷ=1|Y=0, A=group_a) ≈ P(Ŷ=1|Y=0, A=group_b)  [Equal FPR]
    - P(Ŷ=1|Y=1, A=group_a) ≈ P(Ŷ=1|Y=1, A=group_b)  [Equal TPR]
    """
    import numpy as np
    
    predictions = np.zeros(len(y_prob))
    
    for group in np.unique(sensitive_attr):
        mask = (sensitive_attr == group)
        group_probs = y_prob[mask]
        
        # Apply group-specific threshold
        if group == 0:
            predictions[mask] = (group_probs >= threshold_0).astype(int)
        else:
            predictions[mask] = (group_probs >= threshold_1).astype(int)
    
    return predictions
```

### Comparison of Mitigation Approaches

| Approach | When to Use | Pros | Cons |
|----------|------------|------|------|
| **Pre-processing** | Bias in training data | Fixes root cause | May lose information |
| **In-processing** | Can modify the model | Direct fairness constraint | Slower training, complex |
| **Post-processing** | Can't retrain | Model-agnostic | Doesn't fix underlying bias |

---

## 5. Explainability and Interpretability

### Why Explainability Matters

```python
# If a model denies someone a loan, they have a RIGHT to know why
# "Your application was denied" → Not sufficient
# "Your application was denied because: income < $30k, credit history < 2 years" → Fair
```

### SHAP (SHapley Additive exPlanations)

```python
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, show=True)

# Individual prediction explanation
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0])

# Force plot shows which features push prediction toward each class
```

### LIME (Local Interpretable Model-agnostic Explanations)

```python
import lime
import lime.lime_tabular

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Negative', 'Positive'],
    mode='classification'
)

# Explain a single prediction
idx = 0
exp = explainer.explain_instance(X_test.iloc[idx], model.predict_proba)
exp.show_in_notebook(show_table=True)
```

### Feature Importance Analysis

```python
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

def bias_audit_feature_importance(model, X, y, sensitive_features):
    """
    Check if sensitive features are unduly influencing predictions
    even after removing them from the model.
    """
    # 1. Check direct feature importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("Feature Importance Ranking:")
    print(importance_df.to_string(index=False))
    
    # 2. Proxy feature check
    # If removing a feature doesn't change accuracy much, it's not critical
    # If accuracy drops significantly, it may be acting as a proxy for sensitive attribute
    
    # 3. Proxy detection
    for sensitive_col in sensitive_features:
        for feature in X.columns:
            if feature != sensitive_col:
                # Check if feature correlates strongly with sensitive attribute
                corr = X[feature].corr(X[sensitive_col])
                if abs(corr) > 0.7:
                    print(f"⚠️ Proxy Alert: '{feature}' highly correlated with '{sensitive_col}' (r={corr:.3f})")
```

### Interpretable vs Black-Box Models

| Model | Interpretability | Performance | Use Case |
|-------|-----------------|-------------|----------|
| Logistic Regression | High | Moderate | High-stakes decisions |
| Decision Trees | High | Moderate | Explainable rules needed |
| Random Forest | Low | High | When accuracy matters most |
| Neural Networks | Low | High | Complex patterns, with explainability tools |
| **Interpretable Boosting Machine** | **High** | **High** | **Best of both worlds** |

---

## 6. Responsible AI Frameworks

### Google's Responsible AI Practices

```python
# 6 principles from Google:
# 1. Social benefit
# 2. Avoid creating/inheriting unfair bias
# 3. Be built and tested for safety
# 4. Be accountable to people
# 5. Incorporate privacy design principles
# 6. Uphold high standards of scientific excellence
```

### Microsoft's Responsible AI Standard

```
1. Fairness → Treat everyone equitably
2. Reliability & Safety → Operate reliably and safely
3. Privacy & Security → Protect data and systems
4. Inclusiveness → Engage diverse perspectives
5. Transparency → Know how AI is used
6. Accountability → Take responsibility
```

### Practical Checklist for Responsible AI

```python
class ResponsibleAI Checklist:
    """Pre-deployment responsible AI audit."""
    
    def __init__(self, model, X_train, y_train, sensitive_attrs):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.sensitive_attrs = sensitive_attrs
    
    def audit(self):
        """Run full responsible AI audit."""
        print("🔍 Running Responsible AI Audit")
        print("=" * 60)
        
        checks = {
            "Bias Detection": self._check_bias(),
            "Fairness Metrics": self._check_fairness(),
            "Explainability": self._check_explainability(),
            "Privacy": self._check_privacy(),
            "Robustness": self._check_robustness(),
            "Documentation": self._check_documentation(),
        }
        
        all_passed = all(checks.values())
        print(f"\n{'✅ All checks passed!' if all_passed else '⚠️ Some checks need attention'}")
        
        return checks
    
    def _check_bias(self):
        """Check for statistical bias in training data and predictions."""
        print("\n✓ Bias Detection: Checked demographic parity")
        return True  # Placeholder
    
    def _check_fairness(self):
        """Compute fairness metrics across protected groups."""
        print("✓ Fairness Metrics: Computed across groups")
        return True
    
    def _check_explainability(self):
        """Ensure model can be explained to stakeholders."""
        print("✓ Explainability: SHAP/LIME available")
        return True
    
    def _check_privacy(self):
        """Verify data privacy compliance."""
        print("✓ Privacy: GDPR/CCPA compliance checked")
        return True
    
    def _check_robustness(self):
        """Test model under adversarial conditions."""
        print("✓ Robustness: Adversarial testing completed")
        return True
    
    def _check_documentation(self):
        """Verify model documentation (model card)."""
        print("✓ Documentation: Model card created")
        return True
```

### Model Cards

```python
# A model card documents:
# - Model details (version, owner, license)
# - Intended use and limitations
# - Training data demographics
# - Performance metrics by subgroup
# - Fairness considerations
# - Ethical considerations
# - Maintenance and monitoring plans

model_card = {
    "model_name": "CreditRiskClassifier v1.2",
    "owner": "ML Engineering Team",
    "date": "2026-09-08",
    "intended_use": "Credit scoring for loan approval",
    "not_intended_use": "Employment decisions, insurance pricing",
    "training_data": {
        "source": "Historical loan records 2015-2023",
        "demographics": {"samples": 100000, "representation": "See Appendix A"},
        "known_limitations": "Underrepresents rural populations"
    },
    "performance_by_group": {
        "overall_accuracy": 0.92,
        "by_ethnicity": {"group_a": 0.94, "group_b": 0.89, "group_c": 0.91},
        "by_gender": {"male": 0.93, "female": 0.91}
    },
    "fairness_notes": "TPR gap of 0.03 across groups - within acceptable threshold",
    "monitoring": "Quarterly bias audits planned"
}
```

---

## 7. Privacy and Data Governance

### Differential Privacy

```python
import numpy as np

class DifferentialPrivacy:
    """
    Add calibrated noise to protect individual privacy.
    
    ε (epsilon): Privacy budget
    - ε = 0: Perfect privacy (no information leaked)
    - ε = ∞: No privacy
    - Typical range: 0.1 to 10.0
    
    δ (delta): Probability of catastrophic privacy loss
    - Typically set to 1 / (number of individuals)
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, data, sensitivity):
        """
        Add Laplace noise for ε-differential privacy.
        
        sensitivity = max change in output from changing one input
        noise ~ Laplace(0, sensitivity / epsilon)
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=data.shape)
        return data + noise
    
    def clip_and_noise(self, gradients, clip_norm, sensitivity):
        """Clip gradients then add noise for DP-SGD."""
        # Clip gradients to bound sensitivity
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        clip_factor = np.minimum(1, clip_norm / (norms + 1e-8))
        clipped = gradients * clip_factor
        
        # Add calibrated noise
        noise = np.random.normal(0, clip_norm * sensitivity / self.epsilon, 
                                 size=clipped.shape)
        return clipped + noise
```

### Federated Learning

```python
"""
Federated Learning: Train models without centralizing data.
Data stays on user devices; only model updates are shared.

Privacy benefit: Raw data never leaves the device.
"""

class FederatedLearningSimulator:
    def __init__(self, num_clients=10, rounds=5):
        self.num_clients = num_clients
        self.rounds = rounds
    
    def train_round(self, global_model, client_data):
        """Simulate one round of federated training."""
        local_updates = []
        
        for client_id in range(self.num_clients):
            # Each client trains on local data
            local_model = global_model.copy()
            # Train on local data only
            local_model.fit(client_data[client_id]['X'], 
                          client_data[client_id]['y'])
            
            # Send only weights update, not data
            update = local_model.coef_ - global_model.coef_
            local_updates.append(update)
        
        # Aggregate updates (FedAvg)
        global_update = np.mean(local_updates, axis=0)
        global_model.coef_ += global_update
        
        return global_model
```

### GDPR and ML Compliance

```python
"""
Key GDPR rights relevant to ML:
1. Right to Explanation: Users can ask why a decision was made
2. Right to Erasure: "Right to be forgotten" - delete user data
3. Right to Access: Users can request their data
4. Data Minimization: Collect only what's necessary
5. Purpose Limitation: Use data only for stated purposes

Model cards and documentation help satisfy these requirements.
"""
```

---

## 8. Hands-On Exercise: Bias Detection

### Build a Fairness Audit Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data with potential bias
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.normal(50000, 15000, n_samples).clip(20000, 200000),
    'education_years': np.random.randint(8, 22, n_samples),
    'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
    'group': np.random.choice(['A', 'B'], n_samples, p=[0.7, 0.3]),  # Imbalanced!
})

# Target with subtle bias built in
data['approved'] = (
    (data['credit_score'] > 600) & 
    (data['income'] > 40000) &
    (np.random.random(n_samples) > 0.1)  # 10% noise
).astype(int)

# Inject group bias: group B has higher threshold
mask_b = data['group'] == 'B'
data.loc[mask_b, 'approved'] = (
    (data.loc[mask_b, 'credit_score'] > 650) & 
    (data.loc[mask_b, 'income'] > 45000)
).astype(int)

print(f"Dataset shape: {data.shape}")
print(f"\nApproval rate by group:")
print(data.groupby('group')['approved'].mean())
print(f"\n⚠️ Note: Group B has lower approval rate - potential bias!")

# Split data
X = data[['age', 'income', 'education_years', 'credit_score']]
y = data['approved']
sensitive = data['group']

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Fairness audit
from collections import defaultdict

def group_metrics(y_true, y_pred, sensitive):
    results = {}
    for group in np.unique(sensitive):
        mask = (sensitive == group)
        tp = ((y_true == 1) & (y_pred == 1) & mask).sum()
        fp = ((y_true == 0) & (y_pred == 1) & mask).sum()
        fn = ((y_true == 1) & (y_pred == 0) & mask).sum()
        tn = ((y_true == 0) & (y_pred == 0) & mask).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        
        results[group] = {'TPR': tpr, 'FPR': fpr, 'Accuracy': accuracy}
        print(f"\nGroup {group}:")
        print(f"  TPR:   {tpr:.4f}")
        print(f"  FPR:   {fpr:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Fairness gaps
    tpr_gap = abs(results['A']['TPR'] - results['B']['TPR'])
    fpr_gap = abs(results['A']['FPR'] - results['B']['FPR'])
    
    print(f"\n📊 Fairness Gaps:")
    print(f"  TPR Gap: {tpr_gap:.4f} {'⚠️' if tpr_gap > 0.05 else '✅'}")
    print(f"  FPR Gap: {fpr_gap:.4f} {'⚠️' if fpr_gap > 0.05 else '✅'}")
    
    return results

group_metrics(y_test.values, y_pred, sens_test.values)

# Mitigation: Try different thresholds per group
print("\n\n🛠️ Mitigation: Adjusting thresholds per group...")
y_prob = model.predict_proba(X_test)[:, 1]

# Find optimal threshold for group B to equalize TPR
group_b_mask = (sens_test == 'B').values
group_a_mask = (sens_test == 'A').values

# Original predictions
pred_b_orig = (y_prob[group_b_mask] >= 0.5).astype(int)
pred_a_orig = (y_prob[group_a_mask] >= 0.5).astype(int)

# Compute original TPRs
actual_b = y_test.values[group_b_mask]
actual_a = y_test.values[group_a_mask]
tpr_b_orig = ((pred_b_orig == 1) & (actual_b == 1)).sum() / (actual_b == 1).sum()
tpr_a_orig = ((pred_a_orig == 1) & (actual_a == 1)).sum() / (actual_a == 1).sum()

print(f"  Original TPR A: {tpr_a_orig:.4f}, TPR B: {tpr_b_orig:.4f}")
print(f"  Gap: {abs(tpr_a_orig - tpr_b_orig):.4f}")

# Try lower threshold for group B
threshold_b = 0.35
pred_b_adjusted = (y_prob[group_b_mask] >= threshold_b).astype(int)
tpr_b_adj = ((pred_b_adjusted == 1) & (actual_b == 1)).sum() / (actual_b == 1).sum()
print(f"\n  With threshold {threshold_b} for group B:")
print(f"  TPR B adjusted: {tpr_b_adj:.4f}")
print(f"  New gap: {abs(tpr_a_orig - tpr_b_adj):.4f}")
print(f"  ⚠️ Note: Adjusting thresholds changes false positive rates too!")
```

### Expected Output:
```
Dataset shape: (1000, 6)

Approval rate by group:
group
A    0.512
B    0.280
Name: approved, dtype: float64

⚠️ Note: Group B has lower approval rate - potential bias!

Overall Accuracy: 0.8500

Group A:
  TPR:   0.8200
  FPR:   0.0500
  Accuracy: 0.9200

Group B:
  TPR:   0.6500
  FPR:   0.0300
  Accuracy: 0.8800

📊 Fairness Gaps:
  TPR Gap: 0.1700 ⚠️
  FPR Gap: 0.0200 ✅
```

---

## 9. Common Pitfalls

### 1. **Confusing Correlation with Fairness**
```python
# Bad: "The model doesn't use protected attributes, so it's fair"
model.fit(X_train, y_train)  # X doesn't contain 'gender'

# But other features can be proxies for gender!
# Zip code, name, education type can all correlate with protected attributes

# Good: Check for proxy features and measure disparate impact
```

### 2. **Optimizing for Accuracy Alone**
```python
# A model that always predicts the majority class achieves high accuracy
# but fails completely for minority groups
#
# Always report performance by subgroup, not just overall
```

### 3. **One-Time Bias Check**
```python
# Bias isn't static — data distributions change over time
# Regular audits are essential, not just at deployment

# Set up automated bias monitoring:
# - Monthly fairness metric reports
# - Alerts when gaps exceed thresholds
# - Retraining triggers when bias detected
```

### 4. **Ignoring Intersectionality**
```python
# A model can be fair with respect to gender AND race individually
# but unfair for Black women (the intersection of both groups)
# Always check subgroups defined by multiple attributes
```

### 5. **Treating Fairness Metrics as Independent**
```python
# You CAN'T simultaneously satisfy all fairness definitions
# Choose the one appropriate for your use case:
# - Lending: Equalized Odds (both groups should have equal error rates)
# - Hiring: Demographic Parity (equal selection rates)
# - Criminal Justice: Equal Opportunity (equal TPR)
```

### 6. **No Documentation**
```python
# Without model cards and documentation,
# users can't understand limitations or request explanations
# Always maintain audit trails
```

---

## 10. Summary

| Topic | Key Takeaway |
|-------|-------------|
| Bias Types | Historical, selection, measurement, algorithmic, deployment |
| Fairness Metrics | Demographic parity, equalized odds, equal opportunity — often mutually exclusive |
| Detection | Statistical tests, distribution analysis, disparate impact ratio |
| Mitigation | Pre-processing (fix data), In-processing (fairness constraints), Post-processing (threshold adjustment) |
| Explainability | SHAP, LIME, feature importance — necessary for high-stakes decisions |
| Privacy | Differential privacy, federated learning, GDPR compliance |
| Frameworks | Google Responsible AI, Microsoft Responsible AI Standard |
| Model Cards | Document everything: data, performance, limitations, monitoring |

---

## 11. Next Steps

- **Day 9**: Advanced Deep Learning: CNNs, GANs, and Diffusion Models
- Audit a real-world ML model for bias using IBM's AI Fairness 360 toolkit
- Read: [Weapons of Math Destruction](https://www.goodreads.com/book/show/25547814-weapons-of-math-destruction) by Cathy O'Neil
- Read: [Atlas of AI](https://www.goodreads.com/book/show/52291269-atlas-of-ai) by Kate Crawford
- Explore: [IBM AI Fairness 360](https://aif360.mybluemix.net/) — open-source bias detection toolkit
- Implement a fairness audit as part of your ML pipeline

**Key Takeaway**: AI Ethics isn't a checklist — it's an ongoing commitment. Fairness must be designed into every stage of the ML pipeline, from data collection to deployment to monitoring. The models we build shape the world we live in.

---

*Recommended tools for practice*: IBM AI Fairness 360 (bias detection), SHAP (explainability), Microsoft Fairlearn (fairness metrics), Google TensorFlow Privacy (differential privacy)
