# Machine Learning Day 10 Tutorial Notes

After advanced deep learning and responsible AI, we expand into **Reinforcement Learning, Recommendation Systems, and Multimodal AI** — three areas that extend ML beyond supervised learning into interactive decision-making, personalized content delivery, and cross-modal understanding.

---

## Table of Contents
1. [Why Expand Beyond Supervised Learning?](#1-why-expand-beyond-supervised-learning)
2. [Reinforcement Learning Foundations](#2-reinforcement-learning-foundations)
3. [Recommendation Systems](#3-recommendation-systems)
4. [Multimodal AI](#4-multimodal-ai)
5. [Hands-On Exercise: Simple RL Agent](#5-hands-on-exercise-simple-rl-agent)
6. [Evaluation and Deployment Considerations](#6-evaluation-and-deployment-considerations)
7. [Common Pitfalls](#7-common-pitfalls)
8. [Summary](#8-summary)
9. [Next Steps](#9-next-steps)

---

## 1. Why Expand Beyond Supervised Learning?

Supervised learning learns from labeled examples (`x → y`). Real-world systems need:

```
Reinforcement Learning: Learn from interaction (state → action → reward)
Recommendation Systems: Predict user preferences from behavior patterns  
Multimodal AI: Understand and generate across text, image, audio, video
```

These approaches enable:
- **Interactive agents**: Game playing, robotics, autonomous systems
- **Personalization**: Netflix, Spotify, Amazon recommendations  
- **Rich understanding**: Image captioning, visual question answering, text-to-image

---

## 2. Reinforcement Learning Foundations

RL involves an **agent** interacting with an **environment** to maximize cumulative reward.

### Core Components

```
State (s) → Agent → Action (a) → Environment → Reward (r), Next State (s')
```

The agent learns a **policy** π(a|s) that maps states to actions to maximize expected reward.

### Markov Decision Process (MDP)

Formally defined by (S, A, P, R, γ):
- **S**: Set of states
- **A**: Set of actions  
- **P**: Transition probability P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor [0,1]

### Value Functions and Q-Learning

**State-value function**: V^π(s) = E[Σγᵗrₜ | s₀=s, π]  
**Action-value function**: Q^π(s,a) = E[Σγᵗrₜ | s₀=s, a₀=a, π]

**Q-Learning update** (off-policy TD control):
```
Q(s,a) ← Q(s,a) + α[r + γ maxₐ' Q(s',a') - Q(s,a)]
```

### Exploration vs Exploitation

- **Exploration**: Try new actions to discover better rewards
- **Exploitation**: Use known high-reward actions
- **ε-greedy**: With probability ε, random action; else, argmaxₐ Q(s,a)

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(self.q_table[state]))  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
            
    def update(self, state, action, reward, next_state):
        td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
```

---

## 3. Recommendation Systems

Recommendation systems predict user-item interactions to suggest relevant content.

### Collaborative Filtering

**User-based**: Find similar users, recommend what they liked  
**Item-based**: Find similar items, recommend similar to what user liked  
**Matrix Factorization**: Latent factor models (SVD, ALS)

```python
# Simplified user-user collaborative filtering
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm != 0 else 0

def user_based_cf(user_item_matrix, target_user, k=5):
    # Compute similarity with all users
    similarities = []
    for other_user in range(user_item_matrix.shape[0]):
        if other_user != target_user:
            sim = cosine_similarity(
                user_item_matrix[target_user], 
                user_item_matrix[other_user]
            )
            similarities.append((other_user, sim))
    
    # Get top-k similar users
    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    
    # Predict ratings for unseen items
    predictions = {}
    for item in range(user_item_matrix.shape[1]):
        if user_item_matrix[target_user, item] == 0:  # Unseen item
            weighted_sum = 0
            similarity_sum = 0
            for user_id, sim in top_k:
                rating = user_item_matrix[user_id, item]
                if rating > 0:  # User rated this item
                    weighted_sum += sim * rating
                    similarity_sum += abs(sim)
            predictions[item] = weighted_sum / similarity_sum if similarity_sum > 0 else 0
    
    return predictions
```

### Content-Based Filtering

Recommend items similar to those user liked based on item features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.item_features = None
        
    def fit(self, item_descriptions):
        self.item_features = self.tfidf.fit_transform(item_descriptions)
        
    def recommend(self, user_profile, top_n=5):
        user_vector = self.tfidf.transform([user_profile])
        similarities = cosine_similarity(user_vector, self.item_features).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return top_indices
```

### Hybrid Approaches

Combine collaborative and content-based filtering:
- **Weighted hybrid**: α * CF_score + (1-α) * CBF_score
- **Feature augmentation**: Add CF features as input to content model
- **Cascade**: Use one to filter candidates for the other

---

## 4. Multimodal AI

Multimodal models process and relate information from multiple modalities (text, image, audio, video).

### Vision-Language Foundations

**Contrastive Learning**: Align image and text representations in shared space

```python
# Simplified CLIP-style contrastive loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        # Contrastive loss (symmetric)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) / 2
```

### Multimodal Architectures

1. **Early Fusion**: Concatenate raw/modality-specific features early
2. **Late Fusion**: Process modalities separately, combine predictions
3. **Hybrid Fusion**: Multiple fusion points at different levels

### Applications
- **Image Captioning**: CNN encoder + RNN/Transformer decoder
- **Visual Question Answering (VQA)**: Joint embedding of image + question
- **Text-to-Image Generation**: Diffusion models conditioned on text prompts
- **Audio-Visual Speech Recognition**: Combine lip movements + audio

```python
# Simplified image captioning approach
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)  # e.g., ResNet
        self.decoder = DecoderRNN(embed_size, vocab_size, attention_dim)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
```

---

## 5. Hands-On Exercise: Simple RL Agent

Implement a Q-learning agent for the FrozenLake environment.

```python
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Create environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize Q-learning agent
agent = QLearningAgent(n_states, n_actions, lr=0.8, gamma=0.95, epsilon=0.1)

# Training parameters
n_episodes = 1000
max_steps_per_episode = 100

# Track rewards
rewards_per_episode = []

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    for step in range(max_steps_per_episode):
        # Choose action (ε-greedy)
        action = agent.select_action(state)
        
        # Take action
        new_state, reward, done, truncated, info = env.step(action)
        
        # Update Q-table
        agent.update(state, action, reward, new_state)
        
        state = new_state
        total_reward += reward
        
        if done or truncated:
            break
    
    rewards_per_episode.append(total_reward)
    
    # Decay epsilon
    if episode % 100 == 0:
        agent.epsilon *= 0.99

# Calculate and print results
avg_reward = np.mean(rewards_per_episode[-100:])
print(f"Average reward over last 100 episodes: {avg_reward:.3f}")

# Plot learning progress
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning on FrozenLake')
plt.grid(True)
plt.show()

# Test learned policy
state, _ = env.reset()
done = False
print("\nTesting learned policy:")
while not done:
    action = np.argmax(agent.q_table[state])  # Greedy action
    state, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break

env.close()
```

Expected output: Agent learns to navigate from start (S) to goal (G) without falling in holes (H), achieving increasing rewards over episodes.

---

## 6. Evaluation and Deployment Considerations

### Reinforcement Learning Evaluation
- **Learning curves**: Average reward per episode over training
- **Policy evaluation**: Average reward over multiple evaluation episodes
- **Sample efficiency**: How many interactions needed to reach performance threshold
- **Stability**: Variance in performance across different random seeds

### Recommendation Systems Evaluation
- **Accuracy metrics**: Precision@K, Recall@K, NDCG@K, MAP
- **Ranking metrics**: AUC, MRR
- **Beyond accuracy**: Diversity, novelty, serendipity, coverage
- **Business metrics**: CTR, conversion rate, session duration, revenue

### Multimodal AI Evaluation
- **Task-specific metrics**: BLEU, ROUGE, METEOR for captioning; VQA accuracy for VQA
- **Alignment metrics**: Image-text retrieval performance (Recall@K)
- **Generation quality**: Human evaluation, CLIP score for text-to-image
- **Robustness**: Performance under modality missing/noisy conditions

### Deployment Considerations
- **RL**: Simulation-to-real gap, safety constraints, online vs offline learning
- **Recommendations**: Cold start problem, scalability, real-time updating, fairness/bias
- **Multimodal**: Modality synchronization, computational cost, privacy across modalities
- **All**: Monitoring for drift, A/B testing, fallback mechanisms, explainability

---

## 7. Common Pitfalls

### Reinforcement Learning
1. **Exploration issues**: Too little exploration → suboptimal policies; too much → slow learning
2. **Reward shaping**: Poorly designed rewards can lead to unintended behavior
3. **Non-stationary environments**: Changing dynamics invalidate learned policies
4. **Function approximation instability**: Deep RL can diverge without proper techniques (experience replay, target networks)

### Recommendation Systems
1. **Popularity bias**: Recommending only popular items, hurting long-tail discovery
2. **Filter bubbles**: Over-personalization reducing content diversity
3. **Cold start**: Unable to recommend for new users/items
4. **Feedback loops**: Recommendations influencing future behavior, amplifying biases

### Multimodal AI
1. **Modality imbalance**: One modality dominating the learned representation
2. **Temporal misalignment**: Audio-video sync issues in video understanding
3. **Data scarcity**: Paired multimodal data is expensive to collect
4. **Modality missing**: Performance degrades significantly when one modality unavailable

---

## 8. Summary

| Topic | Key Takeaway |
|-------|-------------|
| Reinforcement Learning | Learn from interaction via reward signals; balances exploration/exploitation |
| Recommendation Systems | Predict user preferences using collaborative, content-based, or hybrid approaches |
| Multimodal AI | Jointly process multiple modalities using contrastive learning and fusion strategies |
| Evaluation | Task-specific metrics plus business/user experience measures |
| Deployment | Consider safety, scalability, bias, and monitoring from the start |

---

## 9. Next Steps

- **Day 11**: Large Language Models (LLMs) and Prompt Engineering
- Implement a policy gradient method (REINFORCE) and compare with Q-learning
- Build a hybrid recommendation system using matrix factorization + content features
- Create a simple image-text retrieval system using contrastive learning
- Read: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf) by Sutton & Barto
- Read: [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-4899-7637-6) edited by Ricci et al.
- Explore: [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for RL implementations
- Try: [Microsoft Recommenders](https://github.com/microsoft/recommenders) for recommendation algorithms
- Audit: Check your recommendation system for popularity bias and filter bubble effects

**Key Takeaway**: As ML moves beyond static prediction, we gain powerful tools for interactive systems (RL), personalized experiences (recommendations), and rich understanding (multimodal). Each introduces new challenges in evaluation, safety, and deployment — but enables applications that were impossible with supervised learning alone.

---
*Recommended tools for practice*: Gymnasium (RL environments), Surprise/Python-recsys (recommendations), Hugging Face Transformers (multimodal), Stable Baselines3 (RL algorithms), LightFM (hybrid recommendations)