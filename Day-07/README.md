# Machine Learning Day 7 Tutorial Notes

The day after Transformers, we focus on the essential but often overlooked aspect: **shipping ML models to production**. Most ML engineers spend more time on deployment than on model training — this day fixes that imbalance.

---

## Table of Contents
1. [Why MLOps Matters](#1-why-mlops-matters)
2. [Model Serialization & Persistence](#2-model-serialization--persistence)
3. [Model Serving APIs](#3-model-serving-apis)
4. [Monitoring & Observability](#4-monitoring--observability)
5. [CI/CD for ML](#5-cicd-for-ml)
6. [Edge Deployment & Mobile](#6-edge-deployment--mobile)
7. [Cost Optimization & Scaling](#7-cost-optimization--scaling)
8. [Hands-On Exercise: Production Deployment](#8-hands-on-exercise-production-deployment)
9. [Common Pitfalls](#9-common-pitfalls)
10. [Summary](#10-summary)

---

## 1. Why MLOps Matters

### The Gap Between Research and Production

```
Research:   Train model → 95% accuracy → Paper
Production: Train → Serve → Monitor → Retrain → Update → Scale
```

**MLOps** = DevOps for machine learning. It covers the entire lifecycle of an ML system.

### Core MLOps Concerns

| Concern | Description | Tools |
|---------|-------------|-------|
| Experiment Tracking | Log params, metrics, artifacts | MLflow, W&B, DVC |
| Model Versioning | Track model iterations | DVC, MLflow |
| Serving | Deploy model as API | FastAPI, TorchServe, Triton |
| Monitoring | Detect data drift, performance drops | Evidently, Prometheus |
| CI/CD | Automated testing & deployment | GitHub Actions, Jenkins |
| Infrastructure | Scaling, containerization | Docker, Kubernetes |

---

## 2. Model Serialization & Persistence

### Saving and Loading Models

Different frameworks have different serialization methods:

```python
# === PyTorch ===
import torch

# Method 1: State dict (recommended for production)
torch.save(model.state_dict(), "model_weights.pt")
model.load_state_dict(torch.load("model_weights.pt"))
model.eval()

# Method 2: TorchScript (traced model - no Python needed)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")
loaded = torch.jit.load("model_traced.pt")

# Method 3: Full pickle (includes class definition)
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

```python
# === Scikit-learn ===
import joblib

# Save and load (joblib is more efficient for numpy arrays)
joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

```python
# === ONNX (cross-framework) ===
import torch.onnx

# Export PyTorch to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

# Load ONNX model (can run in C++, JavaScript, etc.)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```

### Comparison

| Format | Pros | Cons |
|--------|------|------|
| `state_dict` | Lightweight, flexible | Need class definition to load |
| `TorchScript` | No Python needed, fast | Harder to debug |
| `pickle` | Full model, easy | Security risks, Python-only |
| `ONNX` | Cross-platform, optimized | Conversion overhead |

---

## 3. Model Serving APIs

### FastAPI: Modern ML Serving

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Load model at startup
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model.eval()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    
    return PredictionResponse(
        sentiment="positive" if pred == 1 else "negative",
        confidence=float(probs[0][pred].item())
    )

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Using TorchServe (PyTorch's Official Server)

```bash
# Package model for TorchServe
torch-model-archiver \
    --model-name sentiment-model \
    --version 1.0 \
    --serialized-file model.pt \
    --handler text_classification_handler \
    --extra-files index_to_name.json

# Start server
torchserve --start --ncs --model-store model_store --models sentiment-model=model.mar

# Query
curl http://localhost:8080/predictions/sentiment-model -T request.json
```

### Flask vs FastAPI Comparison

```python
# Flask (simpler, synchronous)
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": prediction.tolist()})

# FastAPI (async, auto-docs, faster)
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(data: InputSchema):
    # Async support, Pydantic validation
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}
```

**Why FastAPI?** Async support, auto-generated OpenAPI docs at `/docs`, Pydantic validation, faster than Flask.

---

## 4. Monitoring & Observability

### Data Drift Detection

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Compare training data to production data
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
report.save_html("drift_report.html")
```

### Prometheus + Grafana Stack

```python
from prometheus_client import start_http_server, Counter, Histogram

# Define metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests')
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Request latency')

@app.post("/predict")
async def predict(request: PredictionRequest):
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    result = model.predict(request.text)
    
    INFERENCE_LATENCY.observe(time.time() - start_time)
    return result

# Start Prometheus metrics server at port 8001
start_http_server(8001)
```

### Model Performance Monitoring

```python
class ModelMonitor:
    def __init__(self, model, threshold=0.05):
        self.model = model
        self.threshold = threshold
        self.baseline_accuracy = None
        self.drift_detected = False
    
    def check_performance(self, y_true, y_pred):
        """Detect significant performance drops."""
        accuracy = (y_true == y_pred).mean()
        
        if self.baseline_accuracy is None:
            self.baseline_accuracy = accuracy
            return
        
        degradation = self.baseline_accuracy - accuracy
        if degradation > self.threshold:
            self.drift_detected = True
            print(f"⚠️ Performance degradation: {degradation:.2%}")
            print("Consider retraining!")
```

---

## 5. CI/CD for ML

### GitHub Actions for ML Pipelines

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
      
      - name: Train model
        run: |
          python train.py --data-path data/
      
      - name: Evaluate model
        run: |
          python evaluate.py
      
      - name: Save model artifact
        uses: actions/upload-artifact@v4
        with:
          name: model
          path: model/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to cloud
        run: |
          echo "Deploying model..."
```

### MLflow for Experiment Tracking

```python
import mlflow
import mlflow.sklearn

# Start an MLflow run
with mlflow.start_run(run_name="transformer-finetuning"):
    # Log parameters
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("model_type", "bert-base-uncased")
    
    # Train model...
    train_model(...)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)
    mlflow.log_metric("loss", 0.05)
    
    # Log model
    mlflow.sklearn.log_model(model, "sentiment_model")
    
    # View results: http://localhost:5000
    print("Run logged to MLflow")
```

---

## 6. Edge Deployment & Mobile

### PyTorch Mobile / CoreML

```python
# === iOS (CoreML) ===
import coremltools as ct

# Convert PyTorch to CoreML
traced = torch.jit.trace(model, example_input)
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    convert_to="mlprogram"
)
mlmodel.save("model.mlpackage")

# === Android (TensorFlow Lite) ===
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# === ONNX Mobile ===
# ONNX models work across mobile platforms
```

### Model Optimization for Edge

```python
# === Quantization (reduce model size by 4x) ===
import torch

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Post-training static quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

# === Pruning ===
import torch.nn.utils.prune as prune

# Prune 20% of weights in linear layers
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)
```

---

## 7. Cost Optimization & Scaling

### Containerize Your Model with Docker

```dockerfile
# Dockerfile for ML model serving
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ml-model .
docker run -p 8000:8000 ml-model
```

### Inference Cost Comparison

| Approach | Cost/Hour | Latency | Use Case |
|----------|-----------|---------|----------|
| CPU (t3.medium) | $0.04 | 50-200ms | Small models, batch |
| GPU (g4dn.xlarge) | $0.526 | 5-20ms | Large models, real-time |
| Serverless | $0.0000167/req | 100-500ms | Sporadic traffic |
| Edge Device | Free (hardware) | 1-10ms | Mobile, IoT |

### Batch Inference for Cost Savings

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchInferenceService:
    """Process multiple requests together for efficiency."""
    
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.queue = []
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    async def predict_batch(self, texts):
        """Process batch asynchronously."""
        inputs = tokenizer(texts, padding=True, truncation=True, 
                          max_length=128, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return torch.softmax(outputs.logits, dim=-1)
```

---

## 8. Hands-On Exercise: Production Deployment

### Build a Complete ML Pipeline

```python
# train.py - Complete training pipeline
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow tracking
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    # Log everything
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "rf_model")
    
    # Save locally
    joblib.dump(model, "rf_model.joblib")
    
    print(f"Model saved with accuracy: {accuracy:.4f}")
```

```python
# serve.py - FastAPI server
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="ML Production API")

# Load model at startup
model = joblib.load("rf_model.joblib")

class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: Features):
    try:
        input_data = pd.DataFrame([{
            "feature1": features.feature1,
            "feature2": features.feature2,
            "feature3": features.feature3,
            "feature4": features.feature4
        }])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].max()
        
        return {"prediction": int(prediction), "confidence": float(probability)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

```bash
# Run the pipeline
python train.py    # Train and log to MLflow
uvicorn serve:app --host 0.0.0.0 --port 8000  # Serve model

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 5.1, "feature2": 3.5, "feature3": 1.4, "feature4": 0.2}'
```

---

## 9. Common Pitfalls

### 1. **Model Serving Without Versioning**
```python
# Bad: No versioning
model = joblib.load("model.joblib")

# Good: Versioned models
model = joblib.load(f"models/v2.1/model.joblib")
```

### 2. **Ignoring Data Validation**
```python
# Always validate input data before prediction
def validate_input(data):
    if data.isnull().any().any():
        raise ValueError("Input contains null values")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")
    return data
```

### 3. **Forgetting to Set Model to Eval Mode**
```python
# Always set to eval before serving!
model.eval()  # Disables dropout, batch norm updates
```

### 4. **No Health Checks**
```python
@app.get("/health")
def health():
    try:
        # Test a small inference
        test_input = ...
        model.predict(test_input)
        return {"status": "healthy"}
    except Exception:
        return {"status": "unhealthy", "error": "Model inference failed"}
```

### 5. **Ignoring Cold Start**
- First request can be slow due to model loading
- Use pre-warming or serverless with provisioned concurrency

---

## 10. Summary

| Topic | Key Takeaway |
|-------|-------------|
| Serialization | Use `state_dict` + TorchScript for production |
| Serving | FastAPI for flexibility, TorchServe for scale |
| Monitoring | Track data drift and model performance |
| CI/CD | Automated testing and deployment |
| Edge | Quantize and prune for mobile deployment |
| Cost | Batch inference, CPU for small models, GPU for large |

---

## 11. Next Steps

- **Day 8**: Advanced AI Ethics, Bias, and Responsible AI
- Deploy your own model on a cloud platform (AWS/GCP/Azure)
- Set up MLflow experiment tracking for your projects
- Read: [MLOps by Géron](https://www.oreilly.com/library/view/machine-learning-operations/9781492044209/)

**Key Takeaway**: A model is only useful when it's actually running in production. MLOps bridges the gap between research and real-world impact.

---

*Recommended tools for practice*: MLflow (tracking), FastAPI (serving), Docker (containerization), DVC (data versioning)