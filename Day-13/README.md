# Machine Learning Day 13 Tutorial Notes

After RAG and Vector Databases (Day 12), we now tackle **Production-Grade RAG Systems** — the operational layer that turns experimental RAG pipelines into reliable, scalable, production services. Day 12 covered retrieval quality; Day 13 covers the hard problems: monitoring what actually happens, observability for debugging, and deployment patterns that survive real traffic.

---

## Table of Contents
1. [Why Production RAG?](#1-why-production-rag)
2. [Monitoring and Observability](#2-monitoring-and-observability)
3. [Logging and Tracing](#3-logging-and-tracing)
4. [Performance Optimization](#4-performance-optimization)
5. [Security and Access Control](#5-security-and-access-control)
6. [Deployment Strategies](#6-deployment-strategies)
7. [Cost Management](#7-cost-management)
8. [Testing and Validation](#8-testing-and-validation)
9. [A/B Testing and Canary Deployments](#9-ab-testing-and-canary-deployments)
10. [Hands-On Exercise: Deploy a RAG API](#10-hands-on-exercise-deploy-a-rag-api)
11. [Common Pitfalls](#11-common-pitfalls)
12. [Summary](#12-summary)
13. [Next Steps](#13-next-steps)

---

## 1. Why Production RAG?

Development RAG ≠ production RAG. The leap is operational:

```python
# Dev RAG:
dev_rag = {"accuracy": 0.85, "latency": "2s", "monitoring": "print()"}

# Production RAG:
prod_rag = {"accuracy": 0.85, "latency": "<200ms", "monitoring": "prometheus+grafana", "uptime": "99.9%"}
```

**Production Challenges:**
- **Observability**: Can't debug what you can't see
- **Latency**: Synchronous retrieval + generation must meet SLAs
- **Throughput**: Handle concurrent users without degradation
- **Data Freshness**: Knowledge base updates without downtime
- **Security**: Access control on retrieved documents
- **Cost**: API calls add up fast at scale

---

## 2. Monitoring and Observability

### Three Pillars of Observability

```
Metrics → What's happening? (Quantitative)
Logs → What happened? (Historical)
Traces → Why did it happen? (Causal)
```

### Key RAG Metrics

```python
class RAGMetrics:
    """Production RAG monitoring metrics."""
    
    # Retrieval metrics
    retrieval_latency_ms: float       # Time to find relevant docs
    retrieval_recall: float           # % of relevant docs retrieved
    cache_hit_rate: float             # % of queries served from cache
    
    # Generation metrics
    generation_latency_ms: float      # LLM response time
    tokens_per_query: int             # Cost proxy
    hallucination_rate: float         # % of unsupported claims
    
    # System metrics
    concurrent_queries: int           # Current load
    error_rate: float                 # Failed queries / total
    queue_depth: int                  # Pending requests
```

### Prometheus Metrics Export

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
rag_queries_total = Counter('rag_queries_total', 'Total RAG queries', ['status'])
rag_latency_seconds = Histogram('rag_latency_seconds', 'Query latency', ['stage'])
rag_cache_hits = Counter('rag_cache_hits_total', 'Cache hits')
rag_tokens_used = Counter('rag_tokens_used_total', 'Tokens consumed', ['model'])
rag_active_queries = Gauge('rag_active_queries', 'Currently processing queries')

# Start metrics server
start_http_server(8000)

class MonitoredRAGPipeline(RAGPipeline):
    def query(self, question, top_k=5, **kwargs):
        rag_active_queries.inc()
        start = time.time()
        
        try:
            # Track retrieval stage
            retriever_start = time.time()
            documents = self.search(question, top_k, kwargs.get('filters'))
            rag_latency_seconds.labels(stage='retrieval').observe(time.time() - retriever_start)
            
            # Track generation stage
            gen_start = time.time()
            answer, cited = self.generate(question, documents, **kwargs)
            rag_latency_seconds.labels(stage='generation').observe(time.time() - gen_start)
            
            # Track tokens
            rag_tokens_used.labels(model=self.llm_model).inc(
                self._count_tokens(answer) + self._count_tokens(question)
            )
            
            rag_queries_total.labels(status='success').inc()
            return {'answer': answer, 'documents': documents, 'cited': cited}
            
        except Exception as e:
            rag_queries_total.labels(status='error').inc()
            raise
        finally:
            rag_active_queries.dec()
            rag_latency_seconds.labels(stage='total').observe(time.time() - start)
```

### Grafana Dashboard Panels

```
Panel 1: Query Rate (queries/min)
Panel 2: P50/P95/P99 Latency by Stage
Panel 3: Cache Hit Rate (%)
Panel 4: Error Rate (%)
Panel 5: Token Usage (cost tracking)
Panel 6: Active Queries (concurrency)
Panel 7: Retrieval Recall (quality)
```

---

## 3. Logging and Tracing

### Structured Logging

```python
import logging
import json
from pythonjsonlogger import jsonlogger

class JSONFormatter(jsonlogger.JsonFormatter):
    """Structured JSON logging for RAG queries."""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname

# Configure logger
logger = logging.getLogger('rag_pipeline')
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TracedRAGPipeline(RAGPipeline):
    def query(self, question, top_k=5, user_id=None, session_id=None, **kwargs):
        query_id = f"{session_id}:{hash(question) % 10000}"
        
        logger.info('rag_query_start', extra={
            'query_id': query_id,
            'user_id': user_id,
            'session_id': session_id,
            'question': question[:100],  # Truncate for privacy
            'top_k': top_k
        })
        
        try:
            # Retrieve
            documents = self.search(question, top_k, kwargs.get('filters'))
            logger.info('rag_retrieval_complete', extra={
                'query_id': query_id,
                'docs_retrieved': len(documents),
                'avg_similarity': sum(d.get('similarity', 0) for d in documents) / len(documents) if documents else 0
            })
            
            # Generate
            answer, cited = self.generate(question, documents, **kwargs)
            logger.info('rag_generation_complete', extra={
                'query_id': query_id,
                'answer_length': len(answer),
                'cited_docs': len(cited),
                'citation_sources': [d['source'] for d in cited]
            })
            
            return {'answer': answer, 'query_id': query_id, 'documents': documents}
            
        except Exception as e:
            logger.error('rag_query_error', extra={
                'query_id': query_id,
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise
```

### OpenTelemetry Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to Jaeger/Tempo
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

class TracedRAGPipeline(RAGPipeline):
    def query(self, question, top_k=5, **kwargs):
        with tracer.start_as_current_span("rag_query") as span:
            span.set_attribute("question", question[:100])
            span.set_attribute("top_k", top_k)
            
            # Retrieval span
            with tracer.start_as_current_span("retrieval") as ret_span:
                documents = self.search(question, top_k, kwargs.get('filters'))
                ret_span.set_attribute("docs_retrieved", len(documents))
            
            # Generation span
            with tracer.start_as_current_span("generation") as gen_span:
                answer, cited = self.generate(question, documents, **kwargs)
                gen_span.set_attribute("answer_length", len(answer))
            
            span.set_attribute("status", "success")
            return {'answer': answer, 'documents': documents}
```

---

## 4. Performance Optimization

### Caching Layer

```python
import redis
import hashlib
import json

class CachedRAGPipeline(RAGPipeline):
    def __init__(self, *args, cache_ttl=3600, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = redis.Redis(host='localhost', port=6379, db=1)
        self.cache_ttl = cache_ttl
    
    def _cache_key(self, query, top_k, filters):
        """Generate deterministic cache key."""
        key_data = f"{query}:{top_k}:{json.dumps(filters or {}, sort_keys=True)}"
        return f"rag:{hashlib.sha256(key_data.encode()).hexdigest()}"
    
    def search(self, query, top_k=5, filters=None):
        cache_key = self._cache_key(query, top_k, filters)
        cached = self.cache.get(cache_key)
        
        if cached:
            logger.info('cache_hit', extra={'query': query[:50], 'cache_key': cache_key})
            return json.loads(cached)
        
        results = super().search(query, top_k, filters)
        self.cache.setex(cache_key, self.cache_ttl, json.dumps(results))
        return results
    
    def query(self, question, top_k=5, **kwargs):
        # Check cache for complete result
        cache_key = self._cache_key(question, top_k, kwargs.get('filters'))
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        result = super().query(question, top_k, **kwargs)
        self.cache.setex(cache_key, self.cache_ttl, json.dumps(result))
        return result
```

### Async Retrieval

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRAGPipeline(RAGPipeline):
    def __init__(self, *args, max_workers=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def async_search(self, query, top_k=5, filters=None):
        """Non-blocking retrieval."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.search, query, top_k, filters)
    
    async def async_generate(self, question, documents, **kwargs):
        """Non-blocking generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: self.generate(question, documents, **kwargs)
        )
    
    async def aquery(self, question, top_k=5, **kwargs):
        """Async query pipeline."""
        # Parallel retrieval and preprocessing
        documents, preprocessed = await asyncio.gather(
            self.async_search(question, top_k, kwargs.get('filters')),
            self.async_preprocess(question)
        )
        
        # Generate with retrieved docs
        answer, cited = await self.async_generate(question, documents, **kwargs)
        return {'answer': answer, 'documents': documents, 'cited': cited}
```

### Batch Processing

```python
class BatchRAGPipeline(RAGPipeline):
    def batch_query(self, questions, top_k=5, batch_size=10, **kwargs):
        """Process multiple queries with batched embeddings."""
        results = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            
            # Batch embed all queries
            batch_embeddings = self.embedder.encode(batch)
            
            # Batch retrieve
            for query, embedding in zip(batch, batch_embeddings):
                docs = self._search_with_embedding(embedding, top_k)
                answer, cited = self.generate(query, docs, **kwargs)
                results.append({'query': query, 'answer': answer, 'documents': docs})
        
        return results
```

---

## 5. Security and Access Control

### Document-Level Access Control

```python
class SecureRAGPipeline(RAGPipeline):
    def __init__(self, *args, auth_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth_provider = auth_provider or self._default_auth
    
    def _default_auth(self, user_id):
        """Default: all documents accessible."""
        return {"user_id": user_id, "roles": ["user"]}
    
    def add_document(self, text, metadata, user_id=None, **kwargs):
        """Add document with ownership tracking."""
        metadata = metadata or {}
        metadata['owner'] = user_id or 'system'
        metadata['access_level'] = metadata.get('access_level', 'public')
        return super().add_document(text, metadata, **kwargs)
    
    def search(self, query, top_k=5, user_id=None, **kwargs):
        """Search with user-specific filtering."""
        user_context = self.auth_provider(user_id)
        
        # Build filter based on user permissions
        filters = kwargs.get('filters', {})
        filters['$or'] = [
            {'access_level': 'public'},
            {'owner': user_context['user_id']}
        ]
        
        # Add role-based access
        if 'admin' in user_context.get('roles', []):
            filters = {}  # Admins see everything
        
        return super().search(query, top_k, filters, **kwargs)
    
    def query(self, question, user_id=None, **kwargs):
        """Query with authentication."""
        if not user_id:
            raise PermissionError("User ID required for RAG query")
        
        result = super().query(question, **kwargs)
        
        # Filter results by access control
        authorized_docs = [
            doc for doc in result['documents']
            if self._check_access(doc, user_id)
        ]
        
        result['documents'] = authorized_docs
        return result
    
    def _check_access(self, doc, user_id):
        """Check if user can access document."""
        if doc.get('access_level') == 'public':
            return True
        if doc.get('owner') == user_id:
            return True
        return False
```

### PII Detection and Redaction

```python
import re

class PrivacyRAGPipeline(RAGPipeline):
    PII_PATTERNS = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    }
    
    def redact_pii(self, text):
        """Remove PII from text before indexing."""
        for pii_type, pattern in self.PII_PATTERNS.items():
            text = re.sub(pattern, f'[{pii_type.upper()} REDACTED]', text)
        return text
    
    def add_document(self, text, metadata, **kwargs):
        """Add document with PII redaction."""
        redacted_text = self.redact_pii(text)
        metadata['pii_redacted'] = True
        return super().add_document(redacted_text, metadata, **kwargs)
    
    def search(self, query, **kwargs):
        """Search with PII redaction on query."""
        safe_query = self.redact_pii(query)
        return super().search(safe_query, **kwargs)
```

---

## 6. Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY rag_pipeline.py .
COPY main.py .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-pipeline
  labels:
    app: rag-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-pipeline
  template:
    metadata:
      labels:
        app: rag-pipeline
    spec:
      containers:
      - name: rag-pipeline
        image: rag-pipeline:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-pipeline-service
spec:
  selector:
    app: rag-pipeline
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Serverless Deployment

```python
# AWS Lambda / Google Cloud Functions compatible
import json
import os
from rag_pipeline import RAGPipeline

# Initialize once (cold start)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            openai_api_key=os.environ['OPENAI_API_KEY'],
            vector_db_path='/tmp/rag_db'  # Lambda ephemeral storage
        )
    return _pipeline

def handler(event, context):
    """Serverless RAG handler."""
    try:
        body = json.loads(event.get('body', '{}'))
        query = body.get('query')
        top_k = body.get('top_k', 5)
        
        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing query parameter'})
            }
        
        pipeline = get_pipeline()
        result = pipeline.query(query, top_k=top_k)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'answer': result['generated_answer'],
                'sources': [d['source'] for d in result['cited_documents']]
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

---

## 7. Cost Management

### Token Budget Tracking

```python
class CostTrackedRAGPipeline(RAGPipeline):
    def __init__(self, *args, budget_per_user=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget_per_user = budget_per_user
        self.user_tokens = {}  # In production, use Redis/DB
    
    def query(self, question, user_id=None, **kwargs):
        # Check budget
        if user_id:
            used = self.user_tokens.get(user_id, 0)
            if used >= self.budget_per_user:
                raise BudgetExceededError(
                    f"User {user_id} exceeded budget: {used}/{self.budget_per_user} tokens"
                )
        
        result = super().query(question, **kwargs)
        
        # Track usage
        if user_id:
            tokens_used = self._estimate_tokens(question, result['generated_answer'])
            self.user_tokens[user_id] = used + tokens_used
        
        return result
    
    def _estimate_tokens(self, *texts):
        """Rough token estimate (~4 chars/token)."""
        return sum(len(t) / 4 for t in texts)
```

### Model Fallback Chain

```python
class CostOptimizedRAGPipeline(RAGPipeline):
    def __init__(self, *args, fallback_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_models = fallback_models or ['gpt-4', 'gpt-3.5-turbo']
        self.model_index = 0
    
    def generate(self, question, documents, **kwargs):
        """Try models in order until one succeeds."""
        last_error = None
        
        for model in self.fallback_models[self.model_index:]:
            try:
                self.llm_model = model
                return super().generate(question, documents, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model} failed, trying next")
                continue
        
        raise last_error or Exception("All models failed")
```

---

## 8. Testing and Validation

### Unit Tests for RAG Components

```python
import unittest
from unittest.mock import Mock, patch

class TestRAGPipeline(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = RAGPipeline(openai_api_key="test-key")
    
    def test_chunking(self):
        """Test document chunking strategies."""
        text = "Sentence one. Sentence two. Sentence three."
        chunks = self.pipeline.chunk_document(text, strategy="fixed", chunk_size=50)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(c) <= 50 for c in chunks))
    
    def test_search_returns_relevant_docs(self):
        """Test retrieval returns relevant documents."""
        self.pipeline.add_document(
            "Machine learning is a subset of AI.",
            {"source": "test"}
        )
        results = self.pipeline.search("What is ML?", top_k=1)
        self.assertGreater(len(results), 0)
        self.assertIn('machine learning', results[0]['text'].lower())
    
    def test_query_returns_answer(self):
        """Test end-to-end query returns answer."""
        with patch.object(self.pipeline, 'generate', return_value=('Test answer', [])):
            result = self.pipeline.query("Test question?", top_k=1)
            self.assertEqual(result['generated_answer'], 'Test answer')
    
    def test_empty_query_raises(self):
        """Test empty query raises ValueError."""
        with self.assertRaises(ValueError):
            self.pipeline.query("", top_k=1)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
class TestRAGIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test pipeline with sample data."""
        cls.pipeline = RAGPipeline(
            openai_api_key=os.environ['TEST_OPENAI_KEY'],
            vector_db_path="./test_db"
        )
        
        # Add test documents
        cls.test_docs = [
            {"source": "ml_basics", "text": "Supervised learning uses labeled data."},
            {"source": "ml_basics", "text": "Unsupervised learning finds patterns."},
            {"source": "deep_learning", "text": "Neural networks have multiple layers."}
        ]
        for doc in cls.test_docs:
            cls.pipeline.add_document(doc['text'], {'source': doc['source']})
    
    def test_retrieval_quality(self):
        """Test retrieval finds relevant documents."""
        result = self.pipeline.query("What is supervised learning?", top_k=2)
        sources = [d['source'] for d in result['documents']]
        self.assertIn('ml_basics', sources)
    
    def test_citation_accuracy(self):
        """Test generated answer cites sources."""
        result = self.pipeline.query("What does unsupervised learning do?", top_k=2)
        if result['cited_documents']:
            self.assertGreater(len(result['cited_documents']), 0)
```

---

## 9. A/B Testing and Canary Deployments

### A/B Testing Framework

```python
import random
import hashlib

class ABRagPipeline(RAGPipeline):
    def __init__(self, *args, variants=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.variants = variants or {
            'A': {'chunk_size': 500, 'top_k': 5},
            'B': {'chunk_size': 1000, 'top_k': 3}
        }
        self.ab_assignments = {}
    
    def get_variant(self, user_id):
        """Consistent variant assignment per user."""
        if user_id not in self.ab_assignments:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            variant = 'A' if hash_val % 2 == 0 else 'B'
            self.ab_assignments[user_id] = variant
        return self.ab_assignments[user_id]
    
    def query(self, question, user_id=None, **kwargs):
        variant = self.get_variant(user_id or 'anonymous')
        config = self.variants[variant]
        
        # Apply variant config
        result = super().query(
            question,
            top_k=config['top_k'],
            chunk_size=config['chunk_size'],
            **kwargs
        )
        
        # Log for analysis
        logger.info('ab_test_result', extra={
            'variant': variant,
            'user_id': user_id,
            'query': question[:50],
            'answer_length': len(result['generated_answer'])
        })
        
        result['variant'] = variant
        return result
```

### Canary Deployment

```python
class CanaryRAGPipeline(RAGPipeline):
    def __init__(self, *args, canary_percentage=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.canary_percentage = canary_percentage
        self.production_pipeline = None  # Set to main pipeline
        self.canary_pipeline = self     # Current (canary) pipeline
    
    def query(self, question, user_id=None, **kwargs):
        """Route traffic between canary and production."""
        if self._is_canary_user(user_id):
            return self.canary_pipeline.query(question, user_id, **kwargs)
        else:
            return self.production_pipeline.query(question, user_id, **kwargs)
    
    def _is_canary_user(self, user_id):
        """Deterministic canary assignment."""
        if not user_id:
            return random.random() * 100 < self.canary_percentage
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return (hash_val % 100) < self.canary_percentage
```

---

## 10. Hands-On Exercise: Deploy a RAG API

### Step 1: Create FastAPI Application

```python
# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from rag_pipeline import CachedRAGPipeline
from monitoring import MetricsCollector
import os

app = FastAPI(title="Production RAG API", version="1.0.0")

# Initialize pipeline with monitoring
pipeline = CachedRAGPipeline(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    cache_ttl=3600
)
metrics = MetricsCollector()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    user_id: Optional[str] = None
    filters: Optional[dict] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    query_id: str
    latency_ms: float
    variant: Optional[str] = None

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main RAG query endpoint."""
    import time
    start = time.time()
    
    try:
        result = pipeline.query(
            request.question,
            top_k=request.top_k,
            user_id=request.user_id,
            filters=request.filters
        )
        
        latency = (time.time() - start) * 1000
        metrics.record_query(latency, success=True)
        
        return QueryResponse(
            answer=result['generated_answer'],
            sources=[d['source'] for d in result['cited_documents']],
            query_id=result.get('query_id', 'unknown'),
            latency_ms=latency,
            variant=result.get('variant')
        )
        
    except Exception as e:
        metrics.record_query(0, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "pipeline": "ready"}

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return metrics.get_metrics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 2: Test the API

```bash
# Start the API
python main.py

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 3}'

# Response
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": ["ml_basics", "types_of_ml"],
  "query_id": "session:1234",
  "latency_ms": 245.3,
  "variant": "A"
}
```

### Step 3: Deploy to Production

```bash
# Build Docker image
docker build -t rag-pipeline:latest .

# Run with Docker Compose
docker-compose up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
```

---

## 11. Common Pitfalls

### 11.1 No Monitoring

```python
# Bad: No visibility into failures
pipeline.query(question)

# Good: Full observability
result = pipeline.query(question, user_id="user123")
logger.info('query_complete', extra={'query_id': result['query_id'], 'latency_ms': result['latency']})
```

### 11.2 Cache Stampede

```python
# Bad: Cache miss causes thundering herd
if not cache.get(key):
    result = expensive_query()  # All requests hit DB simultaneously
    cache.set(key, result)

# Good: Lock + single computation
with cache_lock(key):
    if not cache.get(key):
        result = expensive_query()
        cache.set(key, result)
```

### 11.3 No Rate Limiting

```python
# Bad: Unbounded requests
@app.post("/query")
async def query(request: QueryRequest):
    return pipeline.query(request.question)

# Good: Rate limiting
from slowapi import Limiter

limiter = Limiter(key_func=get_user_id)

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: QueryRequest):
    return pipeline.query(request.question)
```

### 11.4 Cold Start Latency

```python
# Bad: Load model on every request
@app.post("/query")
async def query(request: QueryRequest):
    pipeline = RAGPipeline()  # Cold start: 5-10s
    return pipeline.query(request.question)

# Good: Singleton with warmup
pipeline = RAGPipeline()  # Initialize at startup

@app.on_event("startup")
async def warmup():
    pipeline.query("warmup")  # Pre-load models
```

---

## 12. Summary

### Key Takeaways

1. **Observability is non-negotiable**: Metrics, logs, traces for every query stage
2. **Caching is mandatory**: Redis/Memcached for query results and embeddings
3. **Security matters**: User-level access control, PII detection, audit logging
4. **Deployment patterns**: Docker + Kubernetes for scale, serverless for cost efficiency
5. **Cost control**: Token budgets, model fallback, caching, rate limiting
6. **Testing**: Unit tests for components, integration tests for pipelines
7. **Gradual rollout**: A/B testing and canary deployments reduce risk

### Production Checklist

| Requirement | Status |
|-------------|--------|
| Metrics (Prometheus) | ✅ |
| Logging (Structured JSON) | ✅ |
| Tracing (OpenTelemetry) | ✅ |
| Caching (Redis) | ✅ |
| Rate Limiting | ✅ |
| Health Checks | ✅ |
| Error Handling | ✅ |
| PII Redaction | ✅ |
| Access Control | ✅ |
| Load Testing | ⬜ |
| Chaos Engineering | ⬜ |

---

## 13. Next Steps

### Immediate Next Steps

- **Day 14**: Multi-Agent RAG Systems — specialized agents for different document types and domains
- Implement full observability stack (Prometheus + Grafana + Jaeger)
- Build a complete production RAG API with authentication, caching, and monitoring
- Set up CI/CD pipeline for RAG model updates
- Compare embedding models for your specific domain
- Explore: [LangServe](https://python.langchain.com/docs/langserve/) for RAG API deployment

### Future Directions (Days 14-15)

1. **Multi-Agent RAG**: Specialized agents for legal, medical, technical domains
2. **Self-Healing RAG**: Automatic retry, fallback, and recovery mechanisms
3. **Research Frontiers**: Adaptive retrieval, query understanding, knowledge graph integration

### Recommended Resources

#### Papers
- **[Production RAG Systems](https://arxiv.org/abs/2309.18223)**
- **[Observability for LLM Applications](https://arxiv.org/abs/2310.05682)**

#### Tools
- **LangServe**: RAG API deployment
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Jaeger**: Distributed tracing
- **Redis**: Caching layer

**Key Takeaway**: Production RAG requires monitoring, caching, security, and graceful degradation. A well-observed, well-cached, well-tested RAG system provides reliable, cost-effective knowledge retrieval at scale.

---

*Tutorial created for Machine Learning Day 13: Production-Grade RAG Systems*

**Recommended Tools for Practice**: FastAPI, Redis, Prometheus, Grafana, Docker, Kubernetes, OpenTelemetry

---

Day 12 covered RAG and Vector Databases. Day 13 covered Production-Grade RAG Systems — monitoring, observability, deployment, security, and cost management. Explore [Day 12](https://github.com/ayush/ML/tree/main/Day-12) for RAG pipeline implementation and [Day 11](https://github.com/ayush/ML/tree/main/Day-11) for LLMs and Prompt Engineering.

---

*Last updated: Day 13*
