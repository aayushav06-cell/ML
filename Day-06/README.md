# Machine Learning Day 6 Tutorial Notes

Building on Day 5's deep learning foundations, today we explore the architectures that revolutionized AI: **Transformers** and **Attention mechanisms**. We'll cover how they replaced RNNs, power modern large language models, and enable generative AI — with practical PyTorch examples.

---

## Table of Contents
1. [The Transformer Revolution](#1-the-transformer-revolution)
2. [Self-Attention Mechanism](#2-self-attention-mechanism)
3. [Multi-Head Attention](#3-multi-head-attention)
4. [Positional Encoding](#4-positional-encoding)
5. [Encoder-Decoder Architecture](#5-encoder-decoder-architecture)
6. [Generative AI: From GPT to Diffusion](#6-generative-ai-from-gpt-to-diffusion)
7. [Vision Transformers (ViT)](#7-vision-transformers-vit)
8. [Practical Implementation](#8-practical-implementation)
9. [Hands-On Exercise](#9-hands-on-exercise)
10. [Common Pitfalls](#10-common-pitfalls)
11. [Summary](#11-summary)

---

## 1. The Transformer Revolution

### Why Transformers Changed Everything

Before 2017, **RNNs + attention** were the standard for sequence tasks. But they had critical flaws:

| Problem | RNN/LSTM/GRU | Transformer |
|---------|--------------|-------------|
| Parallelization | ❌ Sequential processing | ✅ Full parallelization |
| Long sequences | ❌ Vanishing gradients | ✅ Handles 4K+ tokens |
| Training speed | Slow | 10-100x faster |
| Context length | Limited (~256 tokens) | Unlimited (with efficient variants) |

**Key insight**: The attention mechanism computes relationships between *all* positions simultaneously, unlike RNNs that process one step at a time.

### The Core Idea: Attention is All You Need

```
Input Sequence → [Attention Layers] → Output Sequence
```

Each word can directly "pay attention" to any other word, regardless of distance.

---

## 2. Self-Attention Mechanism

### The Math Behind Attention

**Scaled Dot-Product Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q** = Queries (what you're looking for)
- **K** = Keys (what each position offers)
- **V** = Values (the actual information)
- **d_k** = dimension of keys (scaling factor prevents large dot products)

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    Single-head self-attention mechanism.
    
    In self-attention, Q, K, V all come from the same input sequence.
    """
    
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_dim]
        """
        # Project to Q, K, V
        Q = self.query(x)  # [batch, seq_len, embed_dim]
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention
        # Q @ K^T gives attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]
        scores = scores / math.sqrt(self.embed_dim)    # Scale
        
        # Softmax over the key dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Example: 2 words, 4-dim embeddings
x = torch.randn(1, 2, 4)
attn = SelfAttention(4)
output, weights = attn(x)

print("Attention weights (2x2 matrix):")
print(weights[0])
# Row i shows how much position i attends to all positions
```

**Explanation**: The attention weights matrix tells us *how much each word should pay attention to every other word*. Softmax ensures weights sum to 1.

---

## 3. Multi-Head Attention

### Why Multiple Heads?

A single attention head can only capture one type of relationship. Multiple heads allow the model to:
- Track syntactic relationships (subject-verb agreement)
- Identify coreference (pronouns to antecedents)
- Detect semantic roles (who-verb-what)

### Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V (single matrix for all heads)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Project and split into heads
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply to values
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.W_o(output)
        
        return output, attention_weights
```

---

## 4. Positional Encoding

### The Problem

Attention has no notion of order. "The cat sat" and "Sat the cat" are treated identically.

### The Solution: Add Positional Information

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 2i for even indices, 2i+1 for odd indices
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Register as buffer (not a parameter, but saved with model state)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Example usage
pos_enc = PositionalEncoding(64)
x = torch.randn(1, 10, 64)  # 10 tokens
x = pos_enc(x)
```

**Intuition**: Each dimension encodes a different frequency, allowing the model to learn relative positions.

---

## 5. Encoder-Decoder Architecture

### Key Components

- **Encoder**: Reads input sequence, produces contextual representations
- **Decoder**: Generates output sequence one token at a time, can attend to encoder outputs

```
Encoder:  [X1, X2, X3, X4] → [Z1, Z2, Z3, Z4]  (Key/Values for decoder)
Decoder:  [Y1, Y2, ..., Yt] → Yt+1  (each step attends to all encoder outputs)
```

### Self-Attention + Cross-Attention

```python
# Decoder attention has two sources:
# 1. Self-attention (look at previously generated tokens)
# 2. Cross-attention (look at encoder outputs)

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
```

---

## 6. Generative AI: From GPT to Diffusion

### GPT: Autoregressive Language Models

GPT generates text one token at a time, using self-attention to look at all previous tokens:

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained GPT-2
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "The future of machine learning is"
inputs = tokenizer(prompt, return_tensors='pt')

# Generate
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        max_length=100,
        num_return_sequences=1,
        temperature=0.8,  # Higher = more creative
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Diffusion Models: Generating Images

```python
# Using Hugging Face diffusers
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "A machine learning model visualized as a neural network, digital art"
image = pipe(prompt, height=512, width=512).images[0]
image.save("ml_model.png")
```

---

## 7. Vision Transformers (ViT)

### Adapting Transformers to Images

Images are split into patches, flattened, and treated as tokens:

```python
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Load pretrained Vision Transformer
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)

# Preprocess image
img = ... # load your image
preprocess = weights.transforms()
img_tensor = preprocess(img).unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=-1)
    top5 = torch.topk(probs, 5, dim=-1)

print("Top 5 classes:", [weights.meta["categories"][i] for i in top5.indices[0]])
```

---

## 8. Practical Implementation: Building a Simple Transformer

```python
class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and feedforward."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class MiniTransformer(nn.Module):
    """Simple transformer for demonstration."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch, seq, embed_dim]
        x = self.pos_enc(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.fc_out(x)
```

---

## 9. Hands-On Exercise

### Task: Fine-tune BERT for Sentiment Analysis

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb", split="train[:1000]")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Training arguments
training_args = TrainingArguments(
    output_dir='./bert-sentiment',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# Evaluate
text = "This movie was absolutely fantastic! Best film I've seen this year."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=-1).item()
print(f"Sentiment: {'Positive' if pred == 1 else 'Negative'}")
```

---

## 10. Common Pitfalls

### 1. **Memory Usage**
```bash
# ViT and large transformers are memory-hungry
# Use gradient checkpointing for long sequences:
model.gradient_checkpointing_enable()
```

### 2. **Overfitting on Small Data**
```python
# Freeze early layers when data is limited:
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False
```

### 3. **Position Embeddings Don't Extrapolate**
- A model trained on 512 tokens may struggle with longer sequences
- Use ALiBi or Rotary Position Embeddings (RoPE) for generalization

### 4. **Attention Isn't Always Better**
- For small datasets, simpler models often win
- Transformers need lots of data and compute

---

## 11. Summary

| Concept | Key Idea | Code Example |
|---------|----------|--------------|
| Self-Attention | All positions attend to all others | `torch.matmul(Q, K^T)` |
| Multi-Head | Multiple parallel attention heads | `nn.Linear` projections |
| Positional Encoding | Add position info to embeddings | `sin/cos` functions |
| Transformers | Attention-only sequence models | `nn.MultiheadAttention` |
| GPT | Autoregressive generation | `model.generate()` |
| Diffusion | Denoising for images | `StableDiffusionPipeline` |
| ViT | Images as token sequences | `vit_b_16` |

---

## Next Steps

- **Day 7**: Generative AI Deep Dive — GANs, VAEs, Diffusion Models in detail
- Build your own transformer: Implement training from scratch
- Read "The Illustrated Transformer" by Jay Alammar
- Experiment: Fine-tune a model on your own dataset

**Key Takeaway**: Transformers enabled the AI boom by making models scale efficiently. Understanding attention is crucial for modern ML work.

---

*Recommended reading*: ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)