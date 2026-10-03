# Machine Learning Day 14 Tutorial Notes

After Production-Grade RAG Systems (Day 13), we now tackle **Multi-Agent RAG Systems** — the next evolution where multiple specialized agents collaborate to retrieve, reason, and generate answers across complex, multi-domain knowledge bases. Single-agent RAG retrieves from one vector store; multi-agent RAG delegates to domain specialists, each with its own retrieval strategy, and synthesizes their findings into a coherent answer.

---

## Table of Contents
1. [Why Multi-Agent RAG?](#1-why-multi-agent-rag)
2. [Architecture Overview](#2-architecture-overview)
3. [Agent Specialization](#3-agent-specialization)
4. [Agent Communication Patterns](#4-agent-communication-patterns)
5. [Orchestration Layer](#5-orchestration-layer)
6. [Domain-Specific Agents](#6-domain-specific-agents)
7. [Multi-Agent RAG Implementation](#7-multi-agent-rag-implementation)
8. [Routing and Dispatch](#8-routing-and-dispatch)
9. [Result Synthesis](#9-result-synthesis)
10. [Evaluation](#10-evaluation)
11. [Hands-On Exercise: Build a Multi-Agent RAG System](#11-hands-on-exercise-build-a-multi-agent-rag-system)
12. [Common Pitfalls](#12-common-pitfalls)
13. [Summary](#13-summary)
14. [Next Steps](#14-next-styles)

---

## 1. Why Multi-Agent RAG?

Single-agent RAG hits limits with complex queries spanning multiple domains:

```python
# Single-agent RAG fails here:
query = "Compare the FDA approval pathway for the new Alzheimer's drug with the EMA process and estimate market impact"
# Requires: medical knowledge + regulatory knowledge + financial analysis
# One retriever can't cover all three domains well
```

**Multi-Agent RAG solves this by:**
- **Specialization**: Each agent is an expert in one domain
- **Parallel retrieval**: Multiple agents search simultaneously
- **Cross-domain reasoning**: Agents discuss and synthesize findings
- **Fallback**: If one agent fails, others compensate

**When to use multi-agent RAG:**
- Queries spanning multiple domains (legal + medical + financial)
- Enterprise knowledge bases with distinct departments
- Research requiring literature + patents + regulatory docs
- Competitive analysis across industries

---

## 2. Architecture Overview

```
User Query
    │
    ▼
┌─────────────────┐
│  Router/Orchestrator │ ← Routes query to relevant agents
└─────────────────┘
    │
    ├──→ Medical Agent → Medical Knowledge Base
    ├──→ Legal Agent   → Legal Knowledge Base
    ├──→ Financial Agent → Financial Knowledge Base
    └──→ Technical Agent → Technical Documentation
    │
    ▼
┌─────────────────┐
│  Synthesis Agent    │ ← Combines agent responses
└─────────────────┘
    │
    ▼
Final Answer (with citations per domain)
```

**Key Components:**
- **Router**: Classifies query and dispatches to relevant agents
- **Specialist Agents**: Domain-specific retrievers and generators
- **Orchestrator**: Manages agent execution order and dependencies
- **Synthesis Agent**: Merges results into coherent answer

---

## 3. Agent Specialization

Each agent has its own:
- **Knowledge base**: Domain-specific vector store
- **Retrieval strategy**: Optimized for that domain's document structure
- **Generation prompt**: Domain-aware prompt template
- **Evaluation criteria**: Domain-specific quality metrics

```python
class DomainAgent:
    """Base class for domain-specific RAG agents."""
    
    def __init__(self, name: str, knowledge_base, llm, embedding_model):
        self.name = name
        self.knowledge_base = knowledge_base  # Domain-specific vector DB
        self.llm = llm
        self.embedding_model = embedding_model
        self.retrieval_config = {}  # Domain-specific chunking/retrieval params
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Domain-specific retrieval."""
        # Each agent can use different retrieval strategies
        raise NotImplementedError
    
    def generate(self, query: str, documents: list) -> str:
        """Domain-specific answer generation."""
        raise NotImplementedError
    
    def evaluate(self, query: str, result: dict) -> float:
        """Domain-specific quality scoring."""
        raise NotImplementedError
```

### Medical Agent
```python
class MedicalAgent(DomainAgent):
    """Specialist for medical/clinical knowledge."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "medical"
        self.retrieval_config = {
            'chunk_size': 256,      # Smaller chunks for precision
            'overlap': 50,
            'similarity_threshold': 0.75,  # High threshold for safety
        }
        self.safety_filters = ['drug_interaction', 'contraindication', 'dosage']
    
    def retrieve(self, query, top_k=5):
        """Retrieve with medical safety filters."""
        results = self.knowledge_base.search(query, top_k=top_k * 2)
        
        # Filter for safety-critical content
        safe_results = [
            r for r in results
            if not any(filt in r.get('tags', []) for filt in self.safety_filters)
        ]
        
        # Add medical entity extraction
        entities = self._extract_medical_entities(query)
        if entities:
            entity_results = self.knowledge_base.search_entities(entities, top_k=3)
            safe_results.extend(entity_results)
        
        return safe_results[:top_k]
    
    def _extract_medical_entities(self, text):
        """Extract medical entities (drugs, conditions, dosages)."""
        import re
        entities = []
        # Simple regex-based extraction (use spaCy/MedSpaCy in production)
        drug_pattern = r'\b[A-Z][a-z]+(inib|mab|ast|pril|olol|dipine)\b'
        drugs = re.findall(drug_pattern, text)
        entities.extend(drugs)
        return entities
    
    def generate(self, query, documents):
        """Generate with medical disclaimer and citation format."""
        context = self._format_medical_context(documents)
        prompt = f"""As a medical information assistant, provide evidence-based answers using the retrieved documents.
        
IMPORTANT: Always include a disclaimer that this is not medical advice.

Documents:
{context}

Question: {query}

Answer (include citations [Source: XXX] and disclaimer):"""
        
        response = self.llm.generate(prompt)
        return response
```

### Legal Agent
```python
class LegalAgent(DomainAgent):
    """Specialist for legal/regulatory knowledge."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "legal"
        self.retrieval_config = {
            'chunk_size': 512,      # Larger chunks for context
            'overlap': 100,
            'include_precedent': True,
        }
    
    def retrieve(self, query, top_k=5):
        """Retrieve with jurisdiction awareness."""
        # Detect jurisdiction from query
        jurisdiction = self._detect_jurisdiction(query)
        
        # Search with jurisdiction filter
        results = self.knowledge_base.search(
            query,
            top_k=top_k,
            filters={'jurisdiction': jurisdiction} if jurisdiction else None
        )
        
        # Augment with related case law
        if 'case' in query.lower() or 'precedent' in query.lower():
            case_results = self.knowledge_base.search_case_law(query, top_k=2)
            results.extend(case_results)
        
        return results[:top_k]
    
    def _detect_jurisdiction(self, query):
        """Detect legal jurisdiction from query text."""
        jurisdiction_keywords = {
            'EU': ['GDPR', 'European Union', 'EU regulation'],
            'US': ['FDA', 'SEC', 'US law', 'federal'],
            'UK': ['UK law', 'England and Wales', 'UK GDPR'],
        }
        for jur, keywords in jurisdiction_keywords.items():
            if any(kw.lower() in query.lower() for kw in keywords):
                return jur
        return None
```

### Financial Agent
```python
class FinancialAgent(DomainAgent):
    """Specialist for financial/market knowledge."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "financial"
        self.retrieval_config = {
            'chunk_size': 384,
            'overlap': 75,
            'include_tickers': True,
        }
        self.risk_disclaimer = True
    
    def retrieve(self, query, top_k=5):
        """Retrieve financial data with ticker symbol extraction."""
        tickers = self._extract_tickers(query)
        
        results = self.knowledge_base.search(query, top_k=top_k)
        
        # Augment with ticker-specific data
        if tickers:
            ticker_results = self.knowledge_base.search_tickers(tickers, top_k=2)
            results.extend(ticker_results)
        
        return results[:top_k]
    
    def _extract_tickers(self, text):
        """Extract stock ticker symbols."""
        import re
        return re.findall(r'\b[A-Z]{1,5}\b', text)  # Simple pattern
```

---

## 4. Agent Communication Patterns

### 4.1 Sequential Pipeline
```
Query → Agent A → Agent B → Agent C → Synthesis
```
Best for: Queries where each agent builds on the previous one's output.

### 4.2 Parallel Dispatch
```
Query → Agent A → Result A
     → Agent B → Result B
     → Agent C → Result C
              ↓
         Synthesis Agent
```
Best for: Independent domain queries (most common).

### 4.3 Debate/Reflection
```
Query → Agent A (pro) → Agent B (con) → Synthesis
```
Best for: Controversial topics requiring multiple perspectives.

### 4.4 Hierarchical
```
Query → Orchestrator → Sub-Orchestrator A → Agent A1, A2
                              → Sub-Orchestrator B → Agent B1, B2
```
Best for: Large organizations with nested domains.

---

## 5. Orchestration Layer

### Router/Dispatcher
```python
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class AgentDispatch:
    agent_name: str
    query: str
    priority: int
    filters: Dict = None

class RagRouter:
    """Routes queries to appropriate specialist agents."""
    
    def __init__(self, agents: Dict[str, DomainAgent]):
        self.agents = agents
        self.domain_keywords = {
            'medical': ['patient', 'doctor', 'drug', 'treatment', 'clinical', 'symptom', 'diagnosis', 'FDA'],
            'legal': ['law', 'regulation', 'compliance', 'contract', 'litigation', 'court', 'statute', 'GDPR'],
            'financial': ['stock', 'market', 'revenue', 'IPO', 'acquisition', 'dividend', 'earnings', 'SEC'],
            'technical': ['API', 'architecture', 'deployment', 'code', 'scalability', 'latency', 'throughput'],
        }
    
    def route(self, query: str, top_k_per_agent: int = 3) -> List[AgentDispatch]:
        """Route query to relevant agents based on keyword matching + LLM classification."""
        query_lower = query.lower()
        
        # Keyword-based routing (fast path)
        dispatched = []
        for domain, keywords in self.domain_keywords.items():
            if any(kw.lower() in query_lower for kw in keywords):
                dispatched.append(AgentDispatch(
                    agent_name=domain,
                    query=query,
                    priority=2 if any(kw.lower() in query_lower for kw in keywords[:2]) else 1,
                    filters={'domain': domain}
                ))
        
        # LLM-based routing (slow path, for ambiguous queries)
        if not dispatched or len(dispatched) < 2:
            llm_dispatch = self._llm_route(query)
            dispatched.extend(llm_dispatch)
        
        # Sort by priority
        dispatched.sort(key=lambda d: d.priority, reverse=True)
        return dispatched[:4]  # Max 4 agents per query
    
    def _llm_route(self, query: str) -> List[AgentDispatch]:
        """Use LLM to classify query into domains."""
        prompt = f"""Classify this query into one or more domains: medical, legal, financial, technical.
Query: {query}
Return JSON with 'domains' list and 'confidence' per domain."""
        
        response = self.agents['medical'].llm.generate(prompt)  # Use any agent's LLM
        import json
        try:
            classification = json.loads(response)
            return [
                AgentDispatch(agent_name=domain, query=query, priority=1)
                for domain in classification.get('domains', [])
            ]
        except json.JSONDecodeError:
            return [AgentDispatch(agent_name='technical', query=query, priority=1)]
```

---

## 6. Domain-Specific Agents

### Building an Agent Factory
```python
class AgentFactory:
    """Creates domain agents with appropriate configurations."""
    
    AGENT_CONFIGS = {
        'medical': {
            'chunk_size': 256,
            'overlap': 50,
            'similarity_threshold': 0.75,
            'embedding_model': 'medical-embedding-model',
            'reranker': 'medical-reranker',
        },
        'legal': {
            'chunk_size': 512,
            'overlap': 100,
            'similarity_threshold': 0.70,
            'embedding_model': 'legal-embedding-model',
            'reranker': 'legal-reranker',
        },
        'financial': {
            'chunk_size': 384,
            'overlap': 75,
            'similarity_threshold': 0.65,
            'embedding_model': 'financial-embedding-model',
            'reranker': 'financial-reranker',
        },
        'technical': {
            'chunk_size': 1024,
            'overlap': 200,
            'similarity_threshold': 0.60,
            'embedding_model': 'code-embedding-model',
            'reranker': 'code-reranker',
        },
    }
    
    @classmethod
    def create_agent(cls, domain: str, vector_db_base_path: str, llm) -> DomainAgent:
        """Create a domain-specific agent."""
        config = cls.AGENT_CONFIGS.get(domain, cls.AGENT_CONFIGS['technical'])
        
        if domain == 'medical':
            return MedicalAgent(
                name='medical',
                knowledge_base=VectorDB(f"{vector_db_base_path}/medical", config),
                llm=llm,
                embedding_model=config['embedding_model'],
            )
        elif domain == 'legal':
            return LegalAgent(
                name='legal',
                knowledge_base=VectorDB(f"{vector_db_base_path}/legal", config),
                llm=llm,
                embedding_model=config['embedding_model'],
            )
        elif domain == 'financial':
            return FinancialAgent(
                name='financial',
                knowledge_base=VectorDB(f"{vector_db_base_path}/financial", config),
                llm=llm,
                embedding_model=config['embedding_model'],
            )
        else:
            return TechnicalAgent(
                name='technical',
                knowledge_base=VectorDB(f"{vector_db_base_path}/technical", config),
                llm=llm,
                embedding_model=config['embedding_model'],
            )
```

---

## 7. Multi-Agent RAG Implementation

### Complete Implementation
```python
import asyncio
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NO_RESULTS = "no_results"

@dataclass
class AgentResult:
    agent_name: str
    query: str
    documents: List[Dict]
    answer: str
    status: AgentStatus
    latency_ms: float
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)

class MultiAgentRAG:
    """Multi-Agent RAG System with orchestration."""
    
    def __init__(self, agents: Dict[str, DomainAgent], orchestrator_config: Dict = None):
        self.agents = agents
        self.router = RagRouter(agents)
        self.orchestrator_config = orchestrator_config or {
            'max_agents': 4,
            'timeout_seconds': 30,
            'min_confidence': 0.5,
            'enable_debate': False,
        }
        self.synthesis_llm = agents['medical'].llm  # Use strongest LLM for synthesis
    
    async def aquery(self, query: str, user_id: str = None) -> Dict:
        """Async multi-agent query."""
        start_time = time.time()
        
        # Step 1: Route query to relevant agents
        dispatches = self.router.route(query)
        logger.info(f"Routed query to {len(dispatches)} agents: {[d.agent_name for d in dispatches]}")
        
        # Step 2: Parallel retrieval across agents
        retrieval_tasks = {
            dispatch.agent_name: self._retrieve_with_timeout(
                dispatch.agent_name,
                dispatch.query,
                timeout=self.orchestrator_config['timeout_seconds']
            )
            for dispatch in dispatches
        }
        
        retrieval_results = await asyncio.gather(*retrieval_tasks.values(), return_exceptions=True)
        
        # Step 3: Convert results to AgentResult objects
        agent_results = []
        for (agent_name, dispatch), result in zip(retrieval_tasks.items(), retrieval_results):
            if isinstance(result, Exception):
                agent_results.append(AgentResult(
                    agent_name=agent_name,
                    query=dispatch.query,
                    documents=[],
                    answer="",
                    status=AgentStatus.FAILED,
                    latency_ms=0,
                    confidence=0.0
                ))
            else:
                agent_results.append(result)
        
        # Step 4: Filter out failed/no-result agents
        successful_results = [
            r for r in agent_results
            if r.status == AgentStatus.SUCCESS and r.documents
        ]
        
        if not successful_results:
            return self._handle_no_results(query, agent_results, start_time)
        
        # Step 5: Synthesis
        final_answer = await self._synthesize(query, successful_results, user_id)
        
        total_latency = (time.time() - start_time) * 1000
        
        return {
            'query': query,
            'answer': final_answer,
            'agent_results': [
                {
                    'agent': r.agent_name,
                    'status': r.status.value,
                    'documents_retrieved': len(r.documents),
                    'latency_ms': r.latency_ms,
                    'confidence': r.confidence,
                }
                for r in agent_results
            ],
            'total_latency_ms': total_latency,
            'agents_consulted': len(dispatches),
            'agents_successful': len(successful_results),
        }
    
    async def _retrieve_with_timeout(self, agent_name: str, query: str, timeout: float) -> AgentResult:
        """Retrieve with timeout protection."""
        agent = self.agents[agent_name]
        start = time.time()
        
        try:
            # Async wrapper for retrieval
            documents = await asyncio.wait_for(
                asyncio.to_thread(agent.retrieve, query),
                timeout=timeout
            )
            
            # Generate answer for this domain
            answer = await asyncio.to_thread(agent.generate, query, documents)
            
            latency = (time.time() - start) * 1000
            confidence = self._calculate_confidence(documents, answer)
            
            return AgentResult(
                agent_name=agent_name,
                query=query,
                documents=documents,
                answer=answer,
                status=AgentStatus.SUCCESS,
                latency_ms=latency,
                confidence=confidence,
                citations=[doc.get('source', '') for doc in documents]
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Agent {agent_name} timed out after {timeout}s")
            return AgentResult(
                agent_name=agent_name, query=query, documents=[],
                answer="", status=AgentStatus.TIMEOUT, latency_ms=timeout * 1000, confidence=0.0
            )
        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            return AgentResult(
                agent_name=agent_name, query=query, documents=[],
                answer="", status=AgentStatus.FAILED, latency_ms=(time.time() - start) * 1000, confidence=0.0
            )
    
    async def _synthesize(self, query: str, results: List[AgentResult], user_id: str = None) -> str:
        """Synthesize multi-agent results into coherent answer."""
        # Build synthesis prompt
        agent_responses = "\n\n".join([
            f"### {r.agent_name.upper()} Agent (confidence: {r.confidence:.2f})\n"
            f"Retrieved {len(r.documents)} documents\n"
            f"Answer: {r.answer}\n"
            f"Sources: {', '.join(r.citations)}"
            for r in results
        ])
        
        synthesis_prompt = f"""Synthesize answers from multiple domain specialists into one coherent response.

Query: {query}

Specialist Answers:
{agent_responses}

Instructions:
1. Address each domain mentioned in the query
2. Resolve contradictions between agents (note uncertainty)
3. Include per-domain citations [Source: AgentName:DocID]
4. Add a "Confidence by Domain" table at the end
5. If any agent failed, note it but don't hallucinate

Synthesized Answer:"""
        
        return self.synthesis_llm.generate(synthesis_prompt)
    
    def _calculate_confidence(self, documents: List[Dict], answer: str) -> float:
        """Calculate confidence score for agent result."""
        if not documents:
            return 0.0
        
        # Factors: similarity scores, document count, answer length, citation coverage
        avg_similarity = sum(doc.get('similarity', 0) for doc in documents) / len(documents)
        doc_factor = min(len(documents) / 5, 1.0)  # Normalize to 5 docs
        answer_factor = min(len(answer) / 200, 1.0)  # Prefer substantive answers
        
        return (avg_similarity * 0.5 + doc_factor * 0.25 + answer_factor * 0.25)
    
    def _handle_no_results(self, query: str, all_results: List[AgentResult], start_time: float) -> Dict:
        """Handle case where no agent returns results."""
        total_latency = (time.time() - start_time) * 1000
        
        return {
            'query': query,
            'answer': "I couldn't find relevant information across any domain. Please rephrase or try a different query.",
            'agent_results': [
                {
                    'agent': r.agent_name,
                    'status': r.status.value,
                    'documents_retrieved': len(r.documents),
                    'latency_ms': r.latency_ms,
                    'confidence': r.confidence,
                }
                for r in all_results
            ],
            'total_latency_ms': total_latency,
            'agents_consulted': len(all_results),
            'agents_successful': 0,
            'error': 'no_results_across_agents',
        }
```

---

## 8. Routing and Dispatch

### Content-Based Routing
```python
class ContentRouter:
    """Routes based on query content analysis."""
    
    def __init__(self):
        self.domain_classifier = None  # Fine-tuned classifier in production
        self.keyword_map = {
            'medical': ['patient', 'clinical', 'FDA', 'drug', 'treatment', 'diagnosis'],
            'legal': ['law', 'regulation', 'compliance', 'contract', 'GDPR', 'SEC'],
            'financial': ['stock', 'market', 'revenue', 'acquisition', 'IPO', 'earnings'],
            'technical': ['API', 'code', 'architecture', 'deployment', 'scalability'],
        }
    
    def route(self, query: str) -> List[str]:
        """Return list of relevant domain names."""
        query_lower = query.lower()
        matched_domains = []
        
        for domain, keywords in self.keyword_map.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score >= 2:  # At least 2 keyword matches
                matched_domains.append(domain)
        
        # Fallback to LLM classification if no clear match
        if not matched_domains:
            matched_domains = self._classify_with_llm(query)
        
        return matched_domains
    
    def _classify_with_llm(self, query: str) -> List[str]:
        """Use LLM for classification when keywords don't match."""
        prompt = f"""Which domains does this query require?医疗, legal, financial, technical.
Query: {query}
Return comma-separated domain names."""
        # Implementation uses LLM to classify
        return ['technical']  # Default fallback
```

### Hybrid Routing (Keywords + Embeddings)
```python
class HybridRouter(ContentRouter):
    """Combines keyword matching with semantic similarity for routing."""
    
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.domain_centroids = {}  # Pre-computed domain query embeddings
    
    def route(self, query: str, threshold: float = 0.5) -> List[str]:
        """Route using both keywords and semantic similarity."""
        # Keyword-based (fast)
        keyword_domains = super().route(query)
        
        # Semantic-based (slow but accurate)
        query_embedding = self.embedding_model.encode(query)
        semantic_domains = []
        
        for domain, centroid in self.domain_centroids.items():
            similarity = cosine_similarity(query_embedding, centroid)
            if similarity > threshold:
                semantic_domains.append(domain)
        
        # Combine: union of both
        all_domains = set(keyword_domains + semantic_domains)
        
        # If still empty, default to technical
        if not all_domains:
            all_domains = {'technical'}
        
        return list(all_domains)
```

---

## 9. Result Synthesis

### Weighted Synthesis
```python
class WeightedSynthesizer:
    """Synthesizes agent results with confidence-weighted scoring."""
    
    def synthesize(self, query: str, results: List[AgentResult]) -> str:
        """Combine agent answers with confidence weighting."""
        # Sort by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)
        
        # Build weighted context
        weighted_context = []
        for r in results:
            weight = r.confidence
            weighted_context.append(
                f"[Weight: {weight:.2f}] {r.agent_name.upper()}: {r.answer}"
            )
        
        synthesis_prompt = f"""Combine these answers using the confidence weights.
Higher confidence answers should dominate; lower confidence answers fill gaps.

Query: {query}

Weighted Answers:
{chr(10).join(weighted_context)}

Provide a unified answer that cites each domain's contribution."""
        
        return self.llm.generate(synthesis_prompt)
```

### Debate-Based Synthesis
```python
class DebateSynthesizer(WeightedSynthesizer):
    """Agents debate before synthesis for controversial topics."""
    
    def __init__(self, *args, debate_rounds=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.debate_rounds = debate_rounds
    
    async def debate(self, query: str, results: List[AgentResult]) -> str:
        """Agents challenge each other's findings before synthesis."""
        pro_agents = [r for r in results if r.confidence > 0.5]
        con_agents = [r for r in results if r.confidence <= 0.5]
        
        debate_log = []
        for round in range(self.debate_rounds):
            for agent_result in pro_agents:
                challenge = f"Challenge from {agent_result.agent_name}: Is this finding consistent across sources?"
                debate_log.append(challenge)
            
            for agent_result in con_agents:
                rebuttal = f"Rebuttal from {agent_result.agent_name}: What evidence supports this claim?"
                debate_log.append(rebuttal)
        
        # Append debate log to synthesis prompt
        debate_text = "
".join(debate_log)
        synthesis_prompt = f"""Synthesize after considering this debate:

Query: {query}
Debate Log:
{debate_text}

Original Answers:
{chr(10).join([f"{r.agent_name}: {r.answer}" for r in results])}

Final Answer (address debate points):"""
        
        return self.llm.generate(synthesis_prompt)
```

---

## 10. Evaluation

### Multi-Agent Evaluation Metrics
```python
class MultiAgentRAGMetrics:
    """Evaluation for multi-agent RAG systems."""
    
    def evaluate(self, query: str, expected_domains: List[str],
                 results: Dict) -> Dict:
        """Evaluate multi-agent response."""
        metrics = {
            'routing_accuracy': self._check_routing(query, expected_domains, results),
            'coverage': self._check_domain_coverage(expected_domains, results),
            'consistency': self._check_consistency(results),
            'latency': results.get('total_latency_ms', 0),
            'agent_efficiency': self._calculate_efficiency(results),
        }
        
        # Per-agent metrics
        metrics['per_agent'] = {}
        for agent_result in results.get('agent_results', []):
            metrics['per_agent'][agent_result['agent']] = {
                'status': agent_result['status'],
                'confidence': agent_result['confidence'],
                'latency_ms': agent_result['latency_ms'],
                'docs_retrieved': agent_result.get('documents_retrieved', 0),
            }
        
        return metrics
    
    def _check_routing(self, query: str, expected: List[str], results: Dict) -> float:
        """Check if correct agents were consulted."""
        consulted = {r['agent'] for r in results.get('agent_results', [])}
        expected_set = set(expected)
        return len(consulted & expected_set) / len(expected_set) if expected_set else 0
    
    def _check_domain_coverage(self, expected: List[str], results: Dict) -> float:
        """Check if all expected domains were covered."""
        successful = {
            r['agent'] for r in results.get('agent_results', [])
            if r['status'] == 'success' and r.get('documents_retrieved', 0) > 0
        }
        return len(successful & set(expected)) / len(expected) if expected else 0
    
    def _check_consistency(self, results: Dict) -> float:
        """Check consistency between agent answers (0-1)."""
        answers = [r.get('answer', '') for r in results.get('agent_results', []) if r.get('answer')]
        if len(answers) < 2:
            return 1.0  # Single agent, trivially consistent
        
        # Simple consistency check: compare answer overlap
        # In production, use LLM-based consistency scoring
        return 0.85  # Placeholder
    
    def _calculate_efficiency(self, results: Dict) -> float:
        """Calculate agent efficiency (successful queries / total queries)."""
        total = len(results.get('agent_results', []))
        successful = sum(1 for r in results.get('agent_results', [])
                        if r['status'] == 'success')
        return successful / total if total > 0 else 0
```

---

## 11. Hands-On Exercise: Build a Multi-Agent RAG System

### Step 1: Setup
```bash
pip install openai sentence-transformers chromadb fastapi uvicorn
export OPENAI_API_KEY="your-key"
```

### Step 2: Create Domain Knowledge Bases
```python
# knowledge_bases.py
medical_docs = [
    {"source": "medical_fda", "text": "FDA approval pathway for new drugs involves preclinical testing, IND application, clinical trials (Phase I-III), NDA submission, and FDA review. Accelerated approval available for serious conditions with surrogate endpoints."},
    {"source": "medical_alzheimers", "text": "Alzheimer's disease treatments: cholinesterase inhibitors (donepezil, rivastigmine), NMDA antagonists (memantine), and newer monoclonal antibodies (lecanemab, donanemab) targeting amyloid plaques."},
]

legal_docs = [
    {"source": "legal_fda_regulations", "text": "FDA regulations (21 CFR) govern drug approval in the US. EMA regulations (EU) govern European approval. Both require GMP compliance and pharmacovigilance."},
    {"source": "legal_GDPR", "text": "GDPR applies to health data processing. Patient consent required for clinical data use. Right to explanation for automated decisions."},
]

financial_docs = [
    {"source": "finance_pharma_market", "text": "Alzheimer's drug market: $5B global market, growing at 12% CAGR. Key players: Eisai (Leqembi), Biogen (Aduhelm). FDA approval drives stock price surges."},
    {"source": "finance_regulatory_impact", "text": "Regulatory approvals impact pharma stock valuations. EMA vs FDA approval timelines differ by 6-12 months, affecting market entry strategy."},
]
```

### Step 3: Initialize Multi-Agent System
```python
from multi_agent_rag import MultiAgentRAG, AgentFactory

# Create agents
agents = {
    'medical': AgentFactory.create_agent('medical', './vector_dbs', llm),
    'legal': AgentFactory.create_agent('legal', './vector_dbs', llm),
    'financial': AgentFactory.create_agent('financial', './vector_dbs', llm),
}

# Add documents to each agent's knowledge base
for doc in medical_docs:
    agents['medical'].knowledge_base.add_document(doc['text'], {'source': doc['source']})
for doc in legal_docs:
    agents['legal'].knowledge_base.add_document(doc['text'], {'source': doc['source']})
for doc in financial_docs:
    agents['financial'].knowledge_base.add_document(doc['text'], {'source': doc['source']})

# Initialize multi-agent RAG
rag = MultiAgentRAG(agents)
```

### Step 4: Run Multi-Domain Query
```python
# Complex multi-domain query
query = "Compare FDA vs EMA approval timelines for Alzheimer's drugs and estimate market impact"

result = rag.query(query)

print(f"Answer: {result['answer']}")
print(f"Agents consulted: {result['agents_consulted']}")
print(f"Total latency: {result['total_latency_ms']:.0f}ms")

# Per-agent breakdown
for agent_r in result['agent_results']:
    print(f"  {agent_r['agent']}: {agent_r['status']} (confidence: {agent_r['confidence']:.2f})")
```

### Step 5: Evaluate
```python
metrics = MultiAgentRAGMetrics().evaluate(
    query=query,
    expected_domains=['medical', 'legal', 'financial'],
    results=result
)

print(f"Routing accuracy: {metrics['routing_accuracy']:.2f}")
print(f"Domain coverage: {metrics['coverage']:.2f}")
print(f"Consistency: {metrics['consistency']:.2f}")
```

---

## 12. Common Pitfalls

### 12.1 Cascading Failures
```python
# Bad: One agent failure breaks entire pipeline
agent_results = [agent.query(query) for agent in self.agents]  # Sequential, fails fast

# Good: Parallel with timeout and fallback
results = await asyncio.gather(*[agent.aquery(query) for agent in self.agents], return_exceptions=True)
```

### 12.2 Redundant Retrieval
```python
# Bad: All agents retrieve the same information
# Fix: Router dispatches only to relevant agents; cache shared documents

# Good: Smart routing with cache sharing
cache_key = hash(query)
if cache_key in self.shared_cache:
    return self.shared_cache[cache_key]  # Skip retrieval for cached queries
```

### 12.3 Contradictory Answers
```python
# Bad: Present contradictory answers without resolution
# Fix: Synthesis agent must detect and flag contradictions

# Good: Debate-based synthesis with confidence weighting
synthesizer = DebateSynthesizer()
answer = synthesizer.debate(query, results)  # Agents challenge, then synthesize
```

### 12.4 Cost Explosion
```python
# Bad: Query all agents for every question
# Fix: Router filters to relevant agents only; budget per query

# Good: Cost-capped multi-agent query
class CostCappedMultiAgentRAG(MultiAgentRAG):
    def __init__(self, *args, max_cost_per_query=0.50, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_cost_per_query = max_cost_per_query
        self.cost_tracker = CostTracker()
    
    async def aquery(self, query, *args, **kwargs):
        estimated_cost = self._estimate_cost(query)
        if estimated_cost > self.max_cost_per_query:
            # Fall back to single-agent (cheapest relevant domain)
            return await self._fallback_query(query)
        return await super().aquery(query, *args, **kwargs)
```

### 12.5 Latency Accumulation
```python
# Bad: Sequential agent execution
for agent in agents:
    result = agent.query(query)  # Each blocks on previous

# Good: Parallel execution
tasks = [agent.aquery(query) for agent in relevant_agents]
results = await asyncio.gather(*tasks)  # All run concurrently
```

---

## 13. Summary

### Key Takeaways

1. **Multi-agent RAG decomposes complex queries**: Each agent handles its domain expertise
2. **Router dispatches intelligently**: Keywords + embeddings for accurate routing
3. **Parallel execution**: Concurrent retrieval across agents minimizes latency
4. **Synthesis unifies findings**: Weighted or debate-based synthesis produces coherent answers
5. **Fault tolerance**: Timeout + fallback per-agent ensures partial results even on failure
6. **Evaluation is multi-dimensional**: Routing accuracy, domain coverage, consistency, latency
7. **Cost control**: Budget caps and caching prevent runaway token costs

### Architecture Decision Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Router | Keyword + LLM hybrid | Fast for obvious queries, accurate for ambiguous |
| Agent execution | Parallel async | Minimizes latency, handles failures gracefully |
| Synthesis | Weighted + debate options | Adapts to query complexity |
| Caching | Shared + per-agent | Reduces redundant retrieval |
| Cost control | Per-query budget | Prevents runaway costs |

---

## 14. Next Steps

### Immediate Next Steps

- **Day 15**: Self-Healing RAG — automatic retry, fallback, and recovery mechanisms
- Implement full multi-agent system with LangGraph or AutoGen
- Build domain-specific knowledge bases for your organization
- Add caching layer (Redis) for shared documents across agents
- Set up evaluation pipeline with RAGAS or custom metrics
- Explore: [LangGraph](https://langchain.com/docs/langgraph/) for agent orchestration, [AutoGen](https://microsoft.github.io/autogen/) for multi-agent frameworks

### Future Directions (Days 15+)

1. **Self-Healing RAG**: Automatic retry, circuit breakers, graceful degradation
2. **Agent Learning**: Agents improve retrieval based on feedback loops
3. **Research Frontiers**: Multi-modal agent RAG (text + image + table retrieval)

### Recommended Resources

#### Papers
- **[Multi-Agent Systems for RAG](https://arxiv.org/abs/2308.05242)**
- **[Agentic RAG: Beyond Retrieval](https://arxiv.org/abs/2310.12430)**

#### Frameworks
- **LangGraph**: Stateful multi-agent orchestration
- **AutoGen**: Microsoft's multi-agent framework
- **CrewAI**: Role-based multi-agent orchestration

#### Tools
- **Redis**: Shared caching layer
- **Prometheus**: Per-agent metrics
- **LangSmith**: Multi-agent tracing and debugging

**Key Takeaway**: Multi-agent RAG scales single-agent RAG to complex, multi-domain queries by delegating to specialists and synthesizing their findings. The orchestration layer (router + synthesis) is the critical design point — get routing right and the system works; get it wrong and you pay for all agents on every query.

---

*Tutorial created for Machine Learning Day 14: Multi-Agent RAG Systems*

**Recommended Tools for Practice**: LangGraph/AutoGen, OpenAI/Anthropic APIs, ChromaDB, FastAPI, Redis, Prometheus

---

Day 13 covered Production-Grade RAG Systems — monitoring, observability, deployment, security, and cost management. Day 14 covers Multi-Agent RAG Systems — domain specialization, routing, orchestration, synthesis, and evaluation. Explore [Day 13](https://github.com/ayush/ML/tree/main/Day-13) for production RAG patterns and [Day 12](https://github.com/ayush/ML/tree/main/Day-12) for RAG fundamentals.

---
*Last updated: Day 14*