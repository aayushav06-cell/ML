# Machine Learning Day 12 Tutorial Notes

After Large Language Models and Prompt Engineering (Day 11), we now tackle **Retrieval-Augmented Generation (RAG) and Vector Databases** — the bridge between static knowledge and dynamic, grounded AI systems. RAG solves the fundamental limitations of LLMs by bringing in external, up-to-date knowledge through vector similarity search, enabling applications like chatbots, question-answering, and knowledge bases that can handle current information.

---

## Table of Contents
1. [Why RAG?](#1-why-rag)
2. [RAG Architecture Overview](#2-rag-architecture-overview)
3. [Vector Databases Explained](#3-vector-databases-explained)
4. [Embedding Models and Chunking](#4-embedding-models-and-chunking)
5. [Vector Database Options](#5-vector-database-options)
6. [RAG Pipeline Implementation](#6-rag-pipeline-implementation)
7. [Advanced RAG Techniques](#7-advanced-rag-techniques)
8. [Evaluation and Metrics](#8-evaluation-and-metrics)
9. [Hands-On Exercise: Complete RAG Pipeline](#9-hands-on-exercise-complete-rag-pipeline)
10. [Common Pitfalls](#10-common-pitfalls)
11. [Summary](#11-summary)
12. [Next Steps](#12-next-steps)

---

## 1. Why RAG?

LLMs have significant limitations that RAG solves:

```python
# LLM Limitations:
 hallucinations = True                    # Can make up facts
 knowledge_cutoff = "September 2021"     # Limited to training data
 privacy_concerns = True                 # Cannot access private data
 static_knowledge = True                 # Knowledge doesn't update
```

**RAG solves these by:**
- Retrieving external, up-to-date knowledge
- Grounding responses in verifiable sources
- Enabling access to private/corporate data
- Continuously updating knowledge base
- Reducing hallucinations with evidence

```
RAG = (Query + Retriever + Generator) → Answer + Evidence
```

**Key Benefits:**
- **Factual Accuracy**: Responses backed by source documents
- **Current Knowledge**: Access to real-time information
- **Privacy**: Private documents stay in your system
- **Custom Knowledge**: Domain-specific expertise
- **Auditable**: Source tracking for verification

**Real-world applications:**
- **Customer Support**: Answer questions using company docs
- **Legal Research**: Cite case law and regulations
- **Medical Knowledge**: Access latest research and guidelines
- **Financial Services**: Use real-time market data and reports

---

## 2. RAG Architecture Overview

RAG follows a three-stage pipeline:

### Stage 1: Retrieval
```
Query → Text Embedding → Vector Similarity Search → Top-k Documents
```

**Key Components:**
- **Query Processing**: Transform text to vectors (embeddings)
- **Vector Storage**: Store document embeddings in vector database
- **Similarity Search**: Find most relevant documents based on vector distance

### Stage 2: Augmentation
```
Retrieved Docs + Query → Context Preparation → Enhanced Prompt
```

**Context Preparation:**
- Concatenate multiple documents
- Add document metadata (source, relevance score)
- Structure for optimal LLM consumption

### Stage 3: Generation
```
Enhanced Prompt + LLM → Answer + Citation
```

**Generation Guidelines:**
- Use retrieved context to ground responses
- Cite sources when referencing specific information
- Maintain conversational flow while being factual
- Handle cases where context is insufficient or irrelevant

```python
# Example RAG Pipeline
class RAGPipeline:
    def __init__(self, llm, vector_db):
        self.llm = llm
        self.vector_db = vector_db
    
    def query(self, question, top_k=5):
        # Stage 1: Retrieve
        query_embedding = self.embedder.embed(question)
        similar_docs = self.vector_db.similarity_search(query_embedding, k=top_k)
        
        # Stage 2: Augment
        context = self.prepare_context(similar_docs, question)
        
        # Stage 3: Generate
        prompt = self.build_prompt(question, context)
        answer = self.llm.generate(prompt)
        
        return answer, similar_docs
```

---

## 3. Vector Databases Explained

### What Are Vector Databases?

Vector databases store and search high-dimensional vector embeddings using similarity metrics (cosine similarity, dot product, Euclidean distance) instead of keyword matching.

**Key Features:**
- **Vector Storage**: Efficiently store millions of embeddings
- **Similarity Search**: Find vectors closest to a query vector
- **Scalability**: Horizontal scaling for large datasets
- **Metadata Filtering**: Search based on document attributes
- **Efficiency**: Optimized for vector similarity operations

### How Vector Similarity Works

```python
# Cosine similarity (most common for text)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Distance metrics:
# - Cosine similarity: <1 means similar, >-1 means dissimilar
# - Euclidean distance: smaller means more similar
# - Dot product: higher means more similar
```

### Vector Database Architecture

```
Vector Database → Storage Engine → Index Structure → Search Algorithm
├── Inverted Index (for keywords)
├── HNSW (Hierarchical Navigable Small World)
├── IVF (Inverted File)
├── LSH (Locality Sensitive Hashing)
└── Metadata Filter (SQL-like queries)
```

**Why Traditional Databases Fail:**
- Relational DB: No native vector similarity search
- Document DB: Limited to text matching, not semantic similarity
- Key-value stores: No similarity operations

---

## 4. Embedding Models and Chunking

### Text Embeddings

**Purpose**: Convert text of any length into fixed-size vectors (usually 256-3072 dimensions)

**Popular Models:**
- **OpenAI Ada-002**: Good balance of accuracy and speed
- **OpenAI text-embedding-ada-002**: 1536 dimensions
- **Claude 3 embeddings**: Strong semantic understanding
- **BERT-based models**: Fine-tuned for specific domains

```python
# Example embedding function
def embed_text(text, model="text-embedding-ada-002"):
    """Convert text to vector embedding."""
    if model == "text-embedding-ada-002":
        # OpenAI API call
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        return response['data'][0]['embedding']
    elif model == "bge-large-en":
        # Local BGE model
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        with torch.no_grad():
            embeddings = model(**tokenizer(text, padding=True, truncation=True, return_tensors="pt"))
            return embeddings.last_hidden_state[:, 0].numpy()
```

### Document Chunking Strategies

**Purpose**: Split large documents into optimal chunks for retrieval

#### 1. Fixed-Size Chunking
```python
def fixed_size_chunking(text, chunk_size=500, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Allow overlap
    
    return chunks
```

#### 2. Semantic Chunking
```python
def semantic_chunking(text, model="sentence-transformers"):
    """Group semantically similar sentences into chunks."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_embedding = None
    
    for sentence in sentences:
        sentence_embedding = embed_text(sentence)
        
        if current_embedding is None:
            current_chunk = [sentence]
            current_embedding = sentence_embedding
        elif cosine_similarity(sentence_embedding.reshape(1, -1), 
                                 current_embedding.reshape(1, -1)) > 0.7:
            current_chunk.append(sentence)
            current_embedding = np.mean([current_embedding, sentence_embedding], axis=0)
        else:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
            current_embedding = sentence_embedding
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks
```

#### 3. Recursive Chunking
```python
def recursive_chunking(text, max_chunk_size=1000, min_chunk_size=200):
    """Recursively split text, then further split if chunks are too small."""
    if len(text) <= max_chunk_size:
        if len(text) >= min_chunk_size:
            return [text]
        else:
            return []
    
    sentences = text.split('. ')
    midpoint = len(sentences) // 2
    
    left_part = '. '.join(sentences[:midpoint]) + '.'
    right_part = '. '.join(sentences[midpoint:]) + '.'
    
    left_chunks = recursive_chunking(left_part, max_chunk_size, min_chunk_size)
    right_chunks = recursive_chunking(right_part, max_chunk_size, min_chunk_size)
    
    return left_chunks + right_chunks
```

### Chunking Optimization Tips

1. **Semantic vs Fixed**: Semantic chunks are better for context, fixed chunks are simpler
2. **Overlap**: Use 10-20% overlap for better context continuity
3. **Sentence Boundaries**: Split at natural sentence boundaries
4. **Metadata**: Include metadata (source, headers) for better filtering
5. **Balancing**: Don't make chunks too small or too large

---

## 5. Vector Database Options

### 1. Pinecone
```python
import pinecone

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("rag-demo")

# Store documents
vectors = [{
    "id": "doc1",
    "values": embedding_vector,
    "metadata": {
        "text": "Document content",
        "source": "company_docs",
        "date": "2024-01-15"
    }
}]
index.upsert(vectors)

# Query similar documents
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"source": "company_docs"},
    include_metadata=True
)
```

**Pros**: Managed service, excellent similarity search, scalable  
**Cons**: Paid, vendor lock-in

### 2. ChromaDB
```python
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

client = chromadb.Client()
collection = client.create_collection(
    name="documents",
    embedding_function=OpenAIEmbeddingFunction(api_key="your-key")
)

# Store documents
collection.add(
    documents=["Document content", "Another document"],
    metadatas=[{"source": "docs"}, {"source": "news"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["similar documents"],
    n_results=5,
    where={"source": "docs"}
)
```

**Pros**: Free/open-source, local deployment, easy to use  
**Cons**: Limited scalability

### 3. Weaviate
```python
import weaviate
from weaviate.embedded import EmbeddedWeaviate

weaviate_uri = EmbeddedWeaviate().start()
client = weaviate.Client(uri=weaviate_uri)

# Define schema
class Document:
    class MetaClass:
        class Vectorizer:
            text2vec_openai = {}

client.schema.create_class(Document.schema())

# Store with vectorization
client.data_object.create(
    data={"content": "Document text", "source": "company_docs"},
    class_name="Document"
)

# Query
result_set = client.query.get(class_name="Document").with_near_vector(
    {"vector": query_embedding, "certainty": 0.7}
).do()
```

**Pros**: Feature-rich, vector + text filtering, scalable  
**Cons**: Steeper learning curve

### 4. Qdrant
```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(path="./qdrant_storage")

# Create collection
client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Store points
points = [PointStruct(
    id=1,
    vector=embedding_vector,
    payload={"text": "Document content", "source": "docs"}
)]
client.upsert_points(collection_name="documents", points=points)

# Query with filters
search_filter = Filter(must=[
    FieldCondition(key="source", match=MatchValue(value="docs"))
])
results = client.query_points(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=search_filter,
    limit=5
)
```

**Pros**: High performance, feature filtering, memory efficient  
**Cons**: Less mature than Pinecone

### 5. FAISS + PostgreSQL
```python
import faiss
import numpy as np
import psycopg2

# FAISS index for vectors
index = faiss.IndexFlatL2(dimension=1536)

# PostgreSQL connection
conn = psycopg2.connect("host=localhost dbname=rag user=postgres password=secret")

# Store vector in FAISS
index.add(np.array([embedding_vector]))

# Store metadata in PostgreSQL
cursor = conn.cursor()
cursor.execute("""
    INSERT INTO documents (id, content, metadata, embedding_id)
    VALUES (%s, %s, %s, %s)
""", (doc_id, text, json_metadata, faiss_id))
conn.commit()

# Hybrid search: FAISS for similarity + PostgreSQL for filtering
query_vector = np.array([query_embedding])
faiss_distances, faiss_indices = index.search(query_vector, k=10)

# Filter indices through PostgreSQL
valid_ids = [faiss_indices[0][i] for i in range(10)]
cursor.execute("""
    SELECT id, content, metadata
    FROM documents
    WHERE id IN %s AND metadata->>'source' = %s
""", (tuple(valid_ids), "docs"))
results = cursor.fetchall()
```

**Pros**: Full control, cost-effective, mature ecosystems  
**Cons**: More complex setup

### Database Comparison

| Database | Pricing | Scalability | Features | Ease of Use |
|----------|---------|-------------|----------|-------------|
| **Pinecone** | Pay-as-you-go | Excellent | Managed, auto-scaling | High |
| **ChromaDB** | Free | Good | Simple, local | Very High |
| **Weaviate** | Cloud options | Excellent | Rich filtering | Medium |
| **Qdrant** | Free/Cloud | Good | Performance-focused | Medium |
| **FAISS+PG** | Minimal | Excellent | Full control | Low |

**Choosing the Right Database:**
- **Proof of Concept**: ChromaDB (free, fast)
- **Production**: Pinecone or Weaviate (managed, scalable)
- **Cost-Conscious**: FAISS + PostgreSQL (control, minimal cost)
- **Research**: Qdrant (performance, feature-rich)

---

## 6. RAG Pipeline Implementation

### Complete End-to-End Implementation

```python
import os
import json
import openai
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import hashlib

class RAGPipeline:
    def __init__(self, openai_api_key: str,
                 embedding_model: str = "text-embedding-ada-002",
                 llm_model: str = "gpt-4",
                 vector_db_path: str = "./chroma_db"):
        
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_db_path = vector_db_path
        
        # Initialize embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize vector database
        self.vector_db = Client(Settings(persist_directory=vector_db_path))
        self.collection = self.vector_db.get_or_create_collection("documents")
        
        # Initialize OpenAI client
        openai.api_key = openai_api_key
    
    def chunk_document(self, text: str, strategy: str = "semantic",
                      chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into chunks using specified strategy."""
        if strategy == "fixed":
            return self._fixed_chunking(text, chunk_size, overlap)
        elif strategy == "semantic":
            return self._semantic_chunking(text)
        elif strategy == "recursive":
            return self._recursive_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _fixed_chunking(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text) and text[end] not in ['.', '!', '?']:
                next_period = text.find('. ', end)
                if next_period != -1 and next_period - end < 100:
                    end = next_period + 2
                    chunk = text[start:end]
            
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    def _semantic_chunking(self, text: str, similarity_threshold: float = 0.7) -> List[str]:
        """Semantic chunking based on embedding similarity."""
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for sentence in sentences:
            sentence_embedding = self.embedder.encode(sentence)
            
            if current_embedding is None:
                current_chunk = [sentence]
                current_embedding = sentence_embedding
            elif np.cosine_similarity(sentence_embedding.reshape(1, -1),
                                     current_embedding.reshape(1, -1)) > similarity_threshold:
                current_chunk.append(sentence)
                current_embedding = np.mean([current_embedding, sentence_embedding], axis=0)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_embedding = sentence_embedding
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _recursive_chunking(self, text: str, max_size: int = 1000,
                           min_size: int = 200) -> List[str]:
        """Recursive chunking strategy."""
        if len(text) <= max_size:
            if len(text) >= min_size:
                return [text]
            else:
                return []
        
        sentences = text.split('. ')
        for split_point in range(len(sentences) // 2, len(sentences)):
            if len('. '.join(sentences[:split_point])) >= min_size:
                left_part = '. '.join(sentences[:split_point]) + '.'
                right_part = '. '.join(sentences[split_point:]) + '.'
                
                left_chunks = self._recursive_chunking(left_part, max_size, min_size)
                right_chunks = self._recursive_chunking(right_part, max_size, min_size)
                
                return left_chunks + right_chunks
        
        mid_point = len(text) // 2
        return (self._recursive_chunking(text[:mid_point], max_size, min_size) +
                self._recursive_chunking(text[mid_point:], max_size, min_size))
    
    def add_document(self, text: str, metadata: Dict,
                    chunking_strategy: str = "semantic") -> List[str]:
        """Add document to the RAG system."""
        chunks = self.chunk_document(text, chunking_strategy)
        chunk_ids = [f"{metadata.get('source', 'doc')}_{hash(chunk) % 10000}" for chunk in chunks]
        embeddings = self.embedder.encode(chunks)
        
        for chunk_id, chunk, embedding in zip(chunk_ids, chunks, embeddings):
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding.tolist()],
                metadatas=[{**metadata, "text": chunk, "chunk_id": chunk_id}],
                documents=[chunk]
            )
        
        return chunk_ids
    
    def search(self, query: str, top_k: int = 5,
              filters: Dict = None) -> List[Dict]:
        """Search for similar documents."""
        query_embedding = self.embedder.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters,
            include_metadata=True
        )
        
        documents = []
        for metadata, document in zip(results['metadatas'][0], results['documents'][0]):
            doc = {
                'text': metadata.get('text', ''),
                'source': metadata.get('source', ''),
                'date': metadata.get('date', ''),
                'similarity': results['distances'][0][list(results['ids'][0]).index(metadata.get('chunk_id', ''))],
                'metadata': {k: v for k, v in metadata.items() if k not in ['text', 'chunk_id']}
            }
            documents.append(doc)
        
        return documents
    
    def generate(self, query: str, documents: List[Dict],
                 max_tokens: int = 1000, temperature: float = 0.7) -> Tuple[str, List[Dict]]:
        """Generate answer using retrieved documents."""
        context = "\n\n".join([
            f"Source: {doc['source']}\nContent: {doc['text']}"
            for doc in documents
        ])
        
        prompt = f"""
        You are a helpful assistant that answers questions based on the provided documents.
        Use only the information from the documents below to answer the question.
        If the documents don't contain enough information to answer, say so.
        
        Question: {query}
        
        Documents:
        {context}
        
        Answer:
        """
        
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers based only on the provided documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        answer = response.choices[0].message.content
        
        cited_documents = []
        for doc in documents:
            if any(sentence in answer for sentence in doc['text'].split('.')[:3]):
                doc['cited'] = True
                cited_documents.append(doc)
        
        return answer, cited_documents
    
    def query(self, question: str, top_k: int = 5,
             filters: Dict = None, generate_answer: bool = True) -> Dict:
        """Main query interface."""
        documents = self.search(question, top_k, filters)
        
        result = {
            'query': question,
            'documents': documents,
            'generated_answer': None,
            'cited_documents': []
        }
        
        if generate_answer and documents:
            answer, cited_docs = self.generate(question, documents)
            result['generated_answer'] = answer
            result['cited_documents'] = cited_docs
        
        return result
```

### Production-Ready RAG Pipeline

```python
import time
import logging
from typing import Optional
import redis

class ProductionRAGPipeline(RAGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        self.metrics = {
            'queries': 0,
            'avg_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def cached_query(self, query: str, filters: Optional[Dict] = None,
                    top_k: int = 5, cache_ttl: int = 3600) -> Dict:
        """Query with Redis caching."""
        cache_key = f"rag_query:{hash(query)}:{json.dumps(filters or {}, sort_keys=True)}"
        
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            self.metrics['cache_hits'] += 1
            self.logger.info(f"Cache hit for query: {query[:50]}...")
            return json.loads(cached_result)
        
        self.metrics['cache_misses'] += 1
        
        start_time = time.time()
        result = self.query(query, top_k, filters)
        response_time = time.time() - start_time
        
        self.metrics['queries'] += 1
        self.metrics['avg_response_time'] = (
            (self.metrics['avg_response_time'] * (self.metrics['queries'] - 1) + response_time) /
            self.metrics['queries']
        )
        
        self.redis_client.setex(cache_key, cache_ttl, json.dumps(result))
        self.logger.info(f"Query executed in {response_time:.2f}s: {query[:50]}...")
        return result
    
    def health_check(self) -> Dict:
        """Check system health."""
        return {
            'vector_db': self._check_vector_db(),
            'embedding_model': self._check_embedding_model(),
            'metrics': self.metrics
        }
    
    def _check_vector_db(self) -> bool:
        try:
            self.collection.query(query_embeddings=[[0] * 384], n_results=1)
            return True
        except Exception as e:
            self.logger.error(f"Vector DB check failed: {e}")
            return False
    
    def _check_embedding_model(self) -> bool:
        try:
            test_embedding = self.embedder.encode(["test"])
            return len(test_embedding) > 0
        except Exception as e:
            self.logger.error(f"Embedding model check failed: {e}")
            return False
```

---

## 7. Advanced RAG Techniques

### 7.1 Hybrid Search

```python
def hybrid_search(self, query: str, top_k: int = 5,
                 filters: Dict = None, alpha: float = 0.5) -> List[Dict]:
    """Combine vector similarity with keyword search."""
    vector_results = self.search(query, top_k, filters)
    keyword_results = self._keyword_search(query, top_k, filters)
    
    combined_results = []
    for doc in vector_results:
        doc['score'] = doc['similarity'] * alpha
        combined_results.append(doc)
    
    for doc in keyword_results:
        existing = next((d for d in combined_results if d['id'] == doc['id']), None)
        if existing:
            existing['score'] += (1 - alpha) * doc['score']
        else:
            doc['score'] = (1 - alpha) * doc['score']
            combined_results.append(doc)
    
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    return combined_results[:top_k]
```

### 7.2 Re-ranking

```python
from transformers import pipeline

class RerankingRAG(RAGPipeline):
    def __init__(self, *args, rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2", **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = pipeline("text-classification", model=rerank_model, tokenizer=rerank_model)
    
    def query_with_reranking(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents using cross-encoder."""
        candidates = [f"{doc['text']}\nSource: {doc['source']}" for doc in documents]
        scores = [self.reranker(f"Query: {query}\nDocument: {c}")[0]['score'] for c in candidates]
        
        for i, doc in enumerate(documents):
            doc['rerank_score'] = (doc['similarity'] + scores[i]) / 2
        
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        return documents[:top_k]
    
    def generate_with_reranking(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        documents = self.search(query, top_k * 2)
        documents = self.query_with_reranking(query, documents, top_k)
        return self.generate(query, documents)
```

### 7.3 Query Expansion

```python
def expand_query(self, query: str, expansion_terms: List[str] = None) -> str:
    """Expand query with synonyms and related terms."""
    if not expansion_terms:
        expansion_terms = self._generate_expansion_terms(query)
    return f"{query} {' '.join(expansion_terms * 2)}"

def _generate_expansion_terms(self, query: str) -> List[str]:
    """Generate related terms using embeddings."""
    query_embedding = self.embedder.encode(query)
    vocabulary = self._get_vocabulary()
    similarities = []
    
    for word in vocabulary:
        if word.lower() in query.lower():
            continue
        word_embedding = self.embedder.encode(word)
        similarity = np.cosine_similarity(
            query_embedding.reshape(1, -1), word_embedding.reshape(1, -1)
        )[0][0]
        similarities.append((word, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in similarities[:5]]
```

### 7.4 Multi-Hop RAG

```python
class MultiHopRAG(RAGPipeline):
    def __init__(self, *args, max_hops: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_hops = max_hops
    
    def multi_hop_query(self, query: str, initial_top_k: int = 5) -> Dict:
        """Multi-hop reasoning across documents."""
        hop_results = []
        current_query = query
        
        for hop in range(self.max_hops):
            result = self.query(current_query, initial_top_k)
            hop_results.append(result)
            
            if hop < self.max_hops - 1:
                current_query = self._generate_followup_query(current_query, result['documents'])
        
        all_documents = []
        seen_ids = set()
        for hop_result in hop_results:
            for doc in hop_result['documents']:
                if doc['id'] not in seen_ids:
                    seen_ids.add(doc['id'])
                    all_documents.append(doc)
        
        final_result = {
            'query': query,
            'all_results': hop_results,
            'combined_documents': all_documents,
            'generated_answer': None,
            'hop_chain': [r['query'] for r in hop_results]
        }
        
        if all_documents:
            final_result['generated_answer'], final_result['cited_docs'] = \
                self.generate(query, all_documents)
        
        return final_result
    
    def _generate_followup_query(self, original_query: str, documents: List[Dict]) -> str:
        if not documents:
            return original_query
        context = "\n".join([f"- {doc['text']}" for doc in documents[:3]])
        prompt = f"Original query: {original_query}\nRetrieved context:\n{context}\nGenerate a follow-up query."
        
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
```

---

## 8. Evaluation and Metrics

### 8.1 Retrieval Metrics

```python
class RAGMetrics:
    def calculate_retrieval_metrics(self, query: str, retrieved_docs: List[Dict],
                                   ground_truth_docs: List[str]) -> Dict:
        """Calculate standard retrieval metrics."""
        metrics = {}
        
        metrics['precision_at_1'] = self._precision_at_k(retrieved_docs, ground_truth_docs, 1)
        metrics['precision_at_3'] = self._precision_at_k(retrieved_docs, ground_truth_docs, 3)
        metrics['precision_at_5'] = self._precision_at_k(retrieved_docs, ground_truth_docs, 5)
        metrics['precision_at_10'] = self._precision_at_k(retrieved_docs, ground_truth_docs, 10)
        
        metrics['recall_at_1'] = self._recall_at_k(retrieved_docs, ground_truth_docs, 1)
        metrics['recall_at_3'] = self._recall_at_k(retrieved_docs, ground_truth_docs, 3)
        metrics['recall_at_5'] = self._recall_at_k(retrieved_docs, ground_truth_docs, 5)
        
        metrics['f1_at_3'] = self._f1_score(metrics['precision_at_3'], metrics['recall_at_3'])
        metrics['map'] = self._mean_average_precision(retrieved_docs, ground_truth_docs)
        metrics['ndcg'] = self._normalized_discounted_cumulative_gain(retrieved_docs, ground_truth_docs)
        
        return metrics
    
    def _precision_at_k(self, retrieved_docs, ground_truth_docs, k):
        if k > len(retrieved_docs): k = len(retrieved_docs)
        relevant = sum(1 for doc in retrieved_docs[:k]
                      if any(gt.lower() in doc['text'].lower() for gt in ground_truth_docs))
        return relevant / k if k > 0 else 0
    
    def _recall_at_k(self, retrieved_docs, ground_truth_docs, k):
        if k > len(retrieved_docs): k = len(retrieved_docs)
        relevant = sum(1 for doc in retrieved_docs[:k]
                      if any(gt.lower() in doc['text'].lower() for gt in ground_truth_docs))
        return relevant / len(ground_truth_docs) if ground_truth_docs else 0
    
    def _f1_score(self, precision, recall):
        if precision + recall == 0: return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def _mean_average_precision(self, retrieved_docs, ground_truth_docs):
        precisions = []
        for k in range(1, len(retrieved_docs) + 1):
            precision_at_k = self._precision_at_k(retrieved_docs, ground_truth_docs, k)
            relevant = any(gt.lower() in doc['text'].lower()
                          for doc in retrieved_docs[:k] for gt in ground_truth_docs)
            if relevant: precisions.append(precision_at_k)
        return np.mean(precisions) if precisions else 0
    
    def _normalized_discounted_cumulative_gain(self, retrieved_docs, ground_truth_docs):
        dcg, idcg = 0, 0
        for i, doc in enumerate(retrieved_docs):
            relevance = 1 if any(gt.lower() in doc['text'].lower() for gt in ground_truth_docs) else 0
            dcg += relevance / np.log2(i + 2)
            if relevance > 0: idcg += 1 / np.log2(i + 2)
        return dcg / idcg if idcg > 0 else 0
```

### 8.2 Generation Metrics

```python
def evaluate_generation(self, answer: str, ground_truth: str, retrieved_docs: List[Dict]) -> Dict:
    """Evaluate generated answer quality."""
    return {
        'factual_correctness': self._calculate_factuality(answer, retrieved_docs),
        'completeness': self._calculate_completeness(answer, ground_truth),
        'conciseness': self._calculate_conciseness(answer, ground_truth),
        'relevance': self._calculate_relevance(answer, ground_truth),
        'citation_accuracy': self._calculate_citation_accuracy(answer, retrieved_docs)
    }

def _calculate_factuality(self, answer, retrieved_docs):
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    supported = sum(1 for s in sentences
                   if any(gt.lower() in s.lower() for doc in retrieved_docs for gt in doc['text'].split('.') if gt.strip()))
    return supported / len(sentences) if sentences else 0

def _calculate_completeness(self, answer, ground_truth):
    answer_words = set(answer.lower().split())
    gt_words = set(ground_truth.lower().split())
    return len(answer_words & gt_words) / len(gt_words) if gt_words else 0

def _calculate_conciseness(self, answer, ground_truth):
    return min(len(answer.split()) / len(ground_truth.split()), 1.0) if ground_truth else 0

def _calculate_relevance(self, answer, ground_truth):
    answer_words = set(answer.lower().split())
    gt_words = set(ground_truth.lower().split())
    return len(answer_words & gt_words) / len(gt_words) if gt_words else 0

def _calculate_citation_accuracy(self, answer, retrieved_docs):
    cited = [doc['source'] for doc in retrieved_docs if doc.get('cited', False)]
    return sum(1 for s in cited if s in answer) / len(cited) if cited else 0
```

### 8.3 Benchmarking Framework

```python
class RAGBenchmark:
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.metrics = RAGMetrics()
        self.results = []
    
    def run_benchmark(self, test_cases):
        benchmark_results = {'overall': {}, 'per_test_case': [], 'retrieval_metrics': {}, 'generation_metrics': {}}
        
        for i, test_case in enumerate(test_cases):
            result = self.rag_pipeline.query(test_case['query'], top_k=10)
            retrieval_metrics = self.metrics.calculate_retrieval_metrics(
                test_case['query'], result['documents'], test_case.get('ground_truth_docs', [])
            )
            generation_metrics = evaluate_generation(
                result.get('generated_answer', ''),
                test_case.get('ground_truth_answer', ''),
                result['documents']
            )
            benchmark_results['per_test_case'].append({
                'test_case_id': i,
                'query': test_case['query'],
                'retrieval_metrics': retrieval_metrics,
                'generation_metrics': generation_metrics
            })
        
        benchmark_results['overall'] = self._aggregate_results(benchmark_results['per_test_case'])
        return benchmark_results
```

---

## 9. Hands-On Exercise: Complete RAG Pipeline

### Step 1: Setup Environment

```bash
pip install openai torch fastapi uvicorn pandas numpy scikit-learn chromadb
export OPENAI_API_KEY="your-api-key"
```

### Step 2: Create Document Collection

```python
ml_documents = [
    {
        "source": "ml_basics",
        "text": "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. ML algorithms build a mathematical model based on training data in order to make predictions or decisions."
    },
    {
        "source": "types_of_ml",
        "text": "There are three main types of machine learning: Supervised Learning, where the model learns from labeled data; Unsupervised Learning, where the model identifies patterns in unlabeled data; and Reinforcement Learning, where the model learns through trial and error with rewards and penalties."
    },
    {
        "source": "deep_learning",
        "text": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers. It excels at recognizing patterns in large amounts of data and is used for image recognition, natural language processing, and speech recognition."
    },
    {
        "source": "neural_networks",
        "text": "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process and transmit information. Common types include Feedforward Neural Networks, Convolutional Neural Networks (CNNs), and Recurrent Neural Networks (RNNs)."
    },
    {
        "source": "model_evaluation",
        "text": "Model evaluation involves assessing how well a machine learning model performs. Key metrics include accuracy, precision, recall, F1-score, and area under the ROC curve (AUC). Cross-validation is used to ensure robust performance estimates."
    },
    {
        "source": "preprocessing",
        "text": "Data preprocessing is crucial for machine learning. It includes steps like data cleaning (handling missing values, removing outliers), feature engineering (creating new features from existing ones), normalization (scaling features), and encoding categorical variables."
    }
]
```

### Step 3: Initialize and Run RAG Pipeline

```python
from rag_pipeline import RAGPipeline

# Initialize
rag_pipeline = RAGPipeline(openai_api_key="your-api-key")

# Add documents
for doc in ml_documents:
    rag_pipeline.add_document(text=doc["text"], metadata={"source": doc["source"]})

# Query
result = rag_pipeline.query("What are the types of machine learning?", top_k=3)
print(result['generated_answer'])
```

### Step 4: Run Evaluation

```python
test_cases = [
    {"query": "What are the main types of machine learning?", "ground_truth_docs": ["types_of_ml"]},
    {"query": "What is deep learning?", "ground_truth_docs": ["deep_learning"]},
    {"query": "What are common metrics for evaluating ML models?", "ground_truth_docs": ["model_evaluation"]}
]

benchmark = RAGBenchmark(rag_pipeline)
results = benchmark.run_benchmark(test_cases)

print(f"Overall Quality: {results['overall']['overall_quality']:.3f}")
print(f"Avg Precision@3: {results['overall']['avg_precision_at_3']:.3f}")
```

---

## 10. Common Pitfalls

### 10.1 Chunking Issues

```python
# Problem: Context window overflow
solution = {"chunk_size": 1000, "overlap": 0, "strategy": "fixed"}

# Fix
solution = {"chunk_size": 512, "overlap": 50, "strategy": "semantic", "min_chunk_size": 100}
```

### 10.2 Embedding Quality

```python
# Problem: Poor semantic similarity
solution = {"embedding_model": "text-embedding-ada-002", "fine_tune_on_domain": True}
```

### 10.3 Retrieval Failures

```python
# Problem: Relevant documents not retrieved
solution = {"pre_filtering": True, "reranking": True, "hybrid_search": True}
```

### 10.4 Generation Issues

```python
# Problem: Hallucinations
solution = {"temperature": 0.1, "max_tokens": 500, "citation_validation": True}
```

### 10.5 Performance and Cost

```python
# Problem: High latency or costs
solution = {"caching": True, "rate_limiting": True, "fallback_models": ["gpt-3.5-turbo"]}
```

---

## 11. Summary

### Key Takeaways

1. **RAG solves core LLM limitations**: Hallucinations, knowledge cutoff, and lack of current information
2. **Vector databases are essential**: Enable semantic similarity search that traditional databases can't provide
3. **Pipeline architecture**: Three-stage process (Retrieval → Augmentation → Generation)
4. **Implementation considerations**: Embedding models, chunking strategies, and database selection
5. **Evaluation is crucial**: Retrieval and generation metrics ensure quality and reliability

### When to Use RAG

| Scenario | Recommendation |
|----------|---------------|
| Chatbots | ✅ Excellent |
| Q&A Systems | ✅ Ideal |
| Customer Support | ✅ Great |
| Research | ✅ Perfect |
| Simple Queries | ⚠️ Overkill |
| No External Data | ❌ Not Needed |

**Key Takeaway**: RAG transforms static LLMs into dynamic knowledge systems that access current, verifiable information through vector similarity search. When properly implemented with good chunking strategies, appropriate embedding models, and effective caching, RAG systems provide accurate, sourced answers that traditional LLMs cannot match.

---

## 12. Next Steps

### Immediate Next Steps

- **Day 13**: Production-Grade RAG Systems — monitoring, observability, and advanced deployment strategies
- Implement a complete RAG pipeline with ChromaDB and OpenAI
- Build a multi-turn chat agent with conversation memory
- Compare prompt engineering vs fine-tuning for a specific task (e.g., sentiment classification)
- Set up OpenAI/Anthropic API with error handling, retries, and cost tracking
- Explore: [LangChain](https://langchain.com/) or [LlamaIndex](https://llamaindex.ai/) for LLM application frameworks

### Future Directions (Days 14-15)

1. **Multi-Agent RAG Systems** with specialized domain agents
2. **Advanced Techniques**: Code-augmented RAG, legal/medical RAG
3. **Research Frontiers**: Retrieval + generation co-training, knowledge graph integration

### Recommended Resources

#### Papers
- **[Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2309.18223)**
- **[Understanding Retrieval-Augmented Generation](https://arxiv.org/abs/2208.05272)**

#### Frameworks
- **LangChain**: Complete RAG framework
- **LlamaIndex**: Alternative with better indexing
- **Haystack**: Open-source QA framework

#### Tools
- **Pinecone**: Managed vector database
- **Weaviate**: Feature-rich vector DB
- **Qdrant**: High-performance vector search
- **ChromaDB**: Simple local vector database

**Key Takeaway**: RAG is the bridge between static LLMs and dynamic, grounded knowledge systems. By combining retrieval, augmentation, and generation, RAG enables applications that are more accurate, current, and trustworthy than standalone LLMs.

---

*Tutorial created for Machine Learning Day 12: Retrieval-Augmented Generation and Vector Databases*

**Recommended Tools for Practice**: OpenAI/Anthropic APIs, ChromaDB/Pinecone, LangChain/LlamaIndex, FastAPI/Uvicorn, Docker

---

Day 11 covered LLMs and Prompt Engineering. Day 13 will cover Production-Grade RAG Systems. Explore [Day 10](https://github.com/ayush/ML/tree/main/Day-10) for Reinforcement Learning, Recommendation Systems, and Multimodal AI.

---
*Last updated: Day 12*
