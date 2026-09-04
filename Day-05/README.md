# Machine Learning Day 5 Tutorial Notes

Building on Days 1-4's foundations, today we dive into the world of deep learning! Deep learning represents a paradigm shift from traditional ML algorithms, enabling us to tackle problems that were previously considered impossible with shallow models.

---

## 1. Introduction to Deep Learning Frameworks

Deep learning frameworks (TensorFlow, PyTorch, JAX) provide the computational infrastructure for building and training neural networks efficiently. They handle complex operations like automatic differentiation, GPU acceleration, and distributed training.

### 1.1 TensorFlow Basics

```python
import tensorflow as tf
from tensorflow import keras

# Create a simple sequential model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

### 1.2 PyTorch Fundamentals

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Initialize model
model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
```

### 1.3 Key Framework Differences

| Feature | TensorFlow | PyTorch | JAX |
|---------|------------|---------|-----|
| Eager Execution | Both (with tf.function) | Immediate | Functional |
| Declarative APIs | Yes (Keras) | Imperative | Functional |
| Mobile/Edge | Excellent (TFLite) | Good (TorchScript) | Limited |
| Ecosystem | Mature (tf.keras) | Research-friendly | Functional ML |
| Debugging | TensorBoard | Python debugging | REPL-friendly |

### 1.4 Choosing Your Framework

```python
# Decision factors:
# Use TensorFlow if:
- You need production deployment (serving, mobile)
- You prefer high-level APIs
- You need extensive documentation

# Use PyTorch if:
- You're doing research or prototyping
- You need maximum flexibility
- You prefer Pythonic debugging

# Use JAX if:
- You're comfortable with functional programming
- You need JIT compilation for speed
- You're implementing novel algorithms
```

---

## 2. Convolutional Neural Networks (CNNs)

CNNs revolutionized computer vision by automatically learning hierarchical visual features. They excel at image processing tasks through convolution operations, pooling, and hierarchical feature extraction.

### 2.1 CNN Architecture Building Blocks

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjust based on input size
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 2.2 Common CNN Architectures

#### ResNet (Residual Networks)
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
```

#### VGGNet
```python
vgg_blocks = []
# VGG uses repeated 3x3 conv blocks
conv_block = [
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
]
vgg_blocks.append(nn.Sequential(*conv_block))
```

### 2.3 Data Augmentation for CNNs

```python
import torchvision.transforms as transforms

# Training augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation/test transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 2.4 CNN Training Tips

```python
# Learning rate scheduling
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# Mixed precision training (if available)
 scaler = torch.cuda.amp.GradScaler()

# Training loop with gradient clipping
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass with gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total
```

---

## 3. Recurrent Neural Networks (RNNs)

RNNs are designed for sequential data where the order of elements matters. They maintain a hidden state that captures information from previous time steps, making them ideal for time series, text, and audio processing.

### 3.1 Simple RNNs

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
```

### 3.2 Long Short-Term Memory (LSTM)

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the last hidden state
        out = self.fc(out[:, -1, :])
        return out

# Training and prediction
model = LSTMModel(input_size=10, hidden_size=256, num_layers=2, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 3.3 Gated Recurrent Unit (GRU)

```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Compare different recurrent cells
models = {
    'RNN': SimpleRNN(10, 256, 2, 3),
    'LSTM': LSTMModel(10, 256, 2, 3),
    'GRU': GRUModel(10, 256, 2, 3)
}

# Training function for any RNN type
def train_rnn(model_name, model, train_loader, criterion, optimizer, device):
    model.to(device)
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = models[model_name](data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'{model_name} - Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### 3.4 Sequence-to-Sequence Models

```python
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Seq2Seq, self).__init__()
        
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, src, trg):
        # Encoding
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Prepare decoder input (start with <SOS> token)
        decoder_input = torch.zeros(trg.size(0), 1, self.hidden_size).to(trg.device)
        
        # Decoding
        decoder_hidden = encoder_hidden
        outputs = []
        
        for t in range(trg.size(1)):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            output = self.fc(decoder_output)
            outputs.append(output)
            
            # Next input is previous output
            decoder_input = output
        
        return torch.stack(outputs, dim=1)
```

---

## 4. Natural Language Processing (NLP)

NLP combines deep learning with language understanding, enabling computers to process, analyze, and generate human language. It leverages pre-trained models and specialized architectures for text processing.

### 4.1 Text Vectorization

```python
import torch
from torchtext.vocab import GloVe, FastText, Vectors
import spacy

# Load spaCy for tokenization and preprocessing
nlp = spacy.load('en_core_web_sm')

def text_to_sequence(text, vocab, max_len=100):
    doc = nlp(text)
    tokens = [vocab.stoi.get(token.text.lower(), vocab.stoi['<unk>']) 
             for token in doc if token.text.lower() in vocab.stoi]
    
    if len(tokens) == 0:
        tokens = [vocab.stoi['<pad>']] * max_len
    elif len(tokens) < max_len:
        tokens += [vocab.stoi['<pad>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return torch.tensor(tokens, dtype=torch.long)

# Load pre-trained vectors
glove_vectors = GloVe(name='6B', dim=300)
fasttext_vectors = FastText(language='en', dimension=300)
```

### 4.2 Word Embeddings

#### Word2Vec
```python
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        return self.proj(embeddings)

# Training Word2Vec
model = Word2Vec(vocab_size=len(vocab), embedding_dim=300)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### BERT-style Embeddings
```python
from transformers import BertTokenizer, BertModel

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        
        return logits

# Using HuggingFace transformers
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertClassifier(n_classes=2)
```

### 4.3 Text Generation

```python
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.5):
        super(TextGenerator, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embeddings(x)
        x = x.unsqueeze(1)  # Reshape for LSTM
        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output.squeeze(1))
        logits = self.fc(output)
        
        return logits, hidden

# Training with teacher forcing
def train_text_generator(model, dataloader, criterion, optimizer, device, tokenizer, max_len):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            # Teacher forcing: feed target as next input
            optimizer.zero_grad()
            
            hidden = None
            logits, hidden = model(input_seq[:, :-1], hidden)
            
            # Shift target sequence for prediction
            targets = target_seq[:, 1:].contiguous().view(-1)
            
            loss = criterion(logits.view(-1, logits.size(-1)), targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
```

### 4.4 Sentiment Analysis

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn as nn

class DistilBERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english',
            num_labels=n_classes
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

# Complete pipeline for sentiment analysis
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df):
    # Add sentiment labels (example)
    df['label'] = df['text'].apply(lambda x: 1 if 'positive' in x.lower() else 0)
    return df

def create_dataloader(df, tokenizer, batch_size=32):
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Tokenize texts
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=128,
        return_tensors='pt'
    )
    
    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

---

## 5. Transfer Learning and Pre-trained Models

Transfer learning leverages pre-trained models on large datasets and adapts them to new tasks with less data and training time.

### 5.1 Image Transfer Learning

```python
import torchvision.models as models
from torchvision import transforms
from torch import nn

# Load pre-trained ResNet
resnet18 = models.resnet18(pretrained=True)

# Modify the final layer for our task
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_classes)

# Create data transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Fine-tuning vs feature extraction
# Option 1: Fine-tune all layers
for param in resnet18.parameters():
    param.requires_grad = True

# Option 2: Only train the final layer (feature extraction)
for param in resnet18.parameters():
    param.requires_grad = False
for param in resnet18.fc.parameters():
    param.requires_grad = True
```

### 5.2 Text Transfer Learning

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load pre-trained RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Training loop
def train_roberta(model, dataloader, optimizer, device):
    model.to(device)
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

# Dataset class for fine-tuning
from torch.utils.data import Dataset

class FineTuneDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }
```

### 5.3 Custom Fine-tuning with PEFT (Parameter-Efficient Fine-Tuning)

```python
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

# LoRA Configuration for parameter-efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # Rank of update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    target_modules=["q_proj", "v_proj"]  # Target attention modules
)

# Apply LoRA to a pre-trained model
base_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model = get_peft_model(base_model, peft_config)

# Print trainable parameters
model.print_trainable_parameters()
# Only 1.8M out of 125M parameters are trainable!
```

---

## 6. Advanced Deep Learning Topics

### 6.1 Adversarial Networks

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )
    
    def forward(self, noise):
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.disc(x)

# Training loop for GAN
def train_gan(gen, disc, gen_optimizer, disc_optimizer, real_data, epochs=100):
    fixed_noise = torch.randn(64, z_dim)
    
    for epoch in range(epochs):
        # Train Discriminator
        disc.train()
        disc_optimizer.zero_grad()
        
        # Real data
        real_loss = disc(real_data)
        real_labels = torch.ones_like(real_loss)
        d_real_loss = nn.BCELoss()(real_loss, real_labels)
        
        # Fake data
        noise = torch.randn(real_data.size(0), z_dim)
        fake_data = gen(noise)
        fake_loss = disc(fake_data.detach())
        fake_labels = torch.zeros_like(fake_loss)
        d_fake_loss = nn.BCELoss()(fake_loss, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        disc_optimizer.step()
        
        # Train Generator
        gen.train()
        gen_optimizer.zero_grad()
        
        fake_data = gen(fixed_noise)
        g_loss = disc(fake_data)
        g_labels = torch.ones_like(g_loss)
        g_loss = nn.BCELoss()(g_loss, g_labels)
        g_loss.backward()
        gen_optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
```

### 6.2 Autoencoders

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Assuming normalized input [0,1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Training for dimensionality reduction and reconstruction
autoencoder = Autoencoder(input_dim=784, hidden_dim=512, latent_dim=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Anomaly detection using reconstruction error
autoencoder.eval()
with torch.no_grad():
    reconstructed, latent = autoencoder(test_data)
    errors = torch.mean((reconstructed - test_data) ** 2, dim=1)
    anomalies = errors > torch.percentile(errors, 95)
```

### 6.3 Reinforcement Learning Basics

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Simple Q-learning environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQN(state_dim, action_dim)
target_agent = DQN(state_dim, action_dim)
target_agent.load_state_dict(agent.state_dict())

target_update_freq = 100
optimizer = optim.Adam(agent.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Experience replay memory
memory = deque(maxlen=10000)

def select_action(state, epsilon=0.1):
    if torch.rand(1) < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent(state_tensor)
            return torch.argmax(q_values).item()

# Training loop
def train_dqn(agent, target_agent, memory, optimizer, criterion, episodes=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            memory.append((state, action, reward, next_state, done))
            
            # Sample from memory and train
            if len(memory) >= 32:
                batch = random.sample(memory, 32)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.FloatTensor(states)
                next_states = torch.FloatTensor(next_states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                dones = torch.BoolTensor(dones).unsqueeze(1)
                
                # Current Q values
                current_q = agent(states).gather(1, actions)
                
                # Next Q values (target network)
                with torch.no_grad():
                    next_q = target_agent(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + (1 - dones.float()) * 0.99 * next_q
                
                loss = criterion(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            
            if episode % 100 == 0:
                print(f'Episode {episode}, Loss: {loss.item():.4f}')
```

---

## 7. Production Deployment

### 7.1 Model Export and Serialization

```python
# Export to TorchScript
 scripted_model = torch.jit.script(model)
scripted_model.save('model.pt')

# Export to ONNX
import onnx
import onnxruntime as ort

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True)

# Validate ONNX model
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)

# Create inference session
sess = ort.InferenceSession('model.onnx')
```

### 7.2 Model Serving

```python
# FastAPI server for model inference
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model globally
model = torch.load('model.pt')
model.eval()

class PredictionRequest(BaseModel):
    input_data: list

@app.post('/predict')
async def predict(request: PredictionRequest):
    try:
        input_tensor = torch.tensor(request.input_data, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        return {'predictions': output.numpy().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health_check():
    return {'status': 'healthy'}
```

### 7.3 Monitoring and Logging

```python
import logging
import wandb
from datetime import datetime

class ModelMonitor:
    def __init__(self, project_name='deep-learning-project'):
        self.wandb = wandb.init(project=project_name)
        self.setup_logging()
    
    def setup_logging(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.basicConfig(
            filename=f'model_training_{timestamp}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, metrics):
        wandb.log(metrics)
        self.logger.info(f"Metrics: {metrics}")
    
    def log_model_checkpoint(self, model, epoch, accuracy):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'wandb_id': self.wandb.id
        }
        torch.save(checkpoint, f'model_checkpoint_epoch_{epoch}.pt')
        wandb.save(f'model_checkpoint_epoch_{epoch}.pt')
        
    def close(self):
        self.wandb.finish()

# Usage in training loop
monitor = ModelMonitor()

for epoch in range(num_epochs):
    # Training...
    train_loss, train_acc = train_epoch(model, train_loader)
    
    # Validation...
    val_loss, val_acc = evaluate_epoch(model, val_loader)
    
    # Log metrics
    monitor.log_metrics({
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'epoch': epoch
    })
    
    # Save checkpoint
    monitor.log_model_checkpoint(model, epoch, val_acc)

monitor.close()
```

---

## 8. Hands-On Exercises

### 8.1 CNN Image Classification

```python
import torchvision
import torchvision.transforms as transforms

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                          download=True, transform=transforms.ToTensor())
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                        download=True, transform=transforms.ToTensor())

# Create a simple CNN model
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Complete pipeline
model = MNISTCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

### 8.2 Text Generation with Transformer

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, block_size):
        super(TransformerGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.decoder = nn.TransformerDecoderLayer(embed_size, num_heads)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
        self.block_size = block_size
    
    def forward(self, x, target):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Generate masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0))
        
        # Decode
        output = self.decoder(target, x, tgt_mask)
        return self.fc_out(output)

# Complete text generation pipeline
# 1. Tokenize data
# 2. Split into train/val
# 3. Create data loaders
# 4. Initialize model
# 5. Train with teacher forcing
# 6. Generate text using beam search or greedy decoding
```

### 8.3 Fine-tuning BERT for Custom Task

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and prepare custom dataset
def load_custom_dataset(filepath):
    df = pd.read_csv(filepath)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Split into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    return train_texts, val_texts, train_labels, val_labels

# Create dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Fine-tuning pipeline
def fine_tune_bert():
    # Load pre-trained tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    train_texts, val_texts, train_labels, val_labels = load_custom_dataset('data.csv')
    
    # Create datasets
    max_len = 128
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False
    )
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            loss = criterion(logits.view(-1), labels.float())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

fine_tune_bert()
```

---

## 9. Common Pitfalls in Deep Learning

### 9.1 Overfitting and How to Prevent It

```python
# Signs of overfitting:
# - Training accuracy much higher than validation accuracy
# - Validation loss starts increasing while training loss decreases
# - Model performs poorly on new, unseen data

# Prevention strategies:
# 1. Early stopping
# 2. Dropout layers
# 3. Data augmentation
# 4. L2 regularization
# 5. Larger training dataset
# 6. Architecture simplification

# Example with early stopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[early_stopping]
)
```

### 9.2 Vanishing and Exploding Gradients

```python
# Solutions:
# 1. Use batch normalization
# 2. Use residual connections (ResNet)
# 3. Use appropriate activation functions (ReLU for hidden layers)
# 4. Initialize weights properly
# 5. Use gradient clipping

# Example with batch normalization and residual connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += residual
        return out
```

### 9.3 Computational Efficiency

```python
# Memory-efficient techniques:
# 1. Mixed precision training
# 2. Gradient accumulation
# 3. Model parallelism (if available)
# 4. Learning rate scheduling
# 5. Proper batch size selection

# Mixed precision training example
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        # Enable mixed precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 9.4 Data Quality Issues

```python
# Data cleaning techniques:
# 1. Handle missing values
# 2. Remove duplicates
# 3. Normalize/Standardize data
# 4. Handle outliers
# 5. Augment data

# Example data cleaning pipeline
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Normalize numerical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Standardize text (lowercase, strip whitespace)
    if 'text' in df.columns:
        df['text'] = df['text'].str.lower().str.strip()
    
    return df
```

---

## 10. Summary for Day 5

1. **Deep Learning Fundamentals** — Understanding neural networks, frameworks (TensorFlow/PyTorch), and the shift from shallow to deep models
2. **Convolutional Neural Networks (CNNs)** — Architecture building blocks, ResNet, VGGNet, data augmentation, and training tips
3. **Recurrent Neural Networks (RNNs)** — Simple RNNs, LSTM, GRU, and sequence-to-sequence models
4. **Natural Language Processing** — Word embeddings, Word2Vec, BERT-style models, and text generation
5. **Transfer Learning** — Fine-tuning pre-trained models, LoRA for parameter-efficient fine-tuning
6. **Advanced Topics** — GANs, autoencoders, and reinforcement learning basics
7. **Production Deployment** — Model export (TorchScript, ONNX), serving with FastAPI, and monitoring
8. **Hands-On Projects** — MNIST classification, text generation, BERT fine-tuning
9. **Common Pitfalls** — Overfitting, vanishing gradients, computational efficiency, and data quality

**Next Steps (Day 6)**:
- Advanced CNN architectures (EfficientNet, MobileNet, YOLO)
- Advanced NLP techniques (BERT, GPT, T5, Llama, Claude)
- Graph neural networks
- AutoML and neural architecture search
- Advanced deployment strategies (GPU/edge deployment, monitoring)
- Real-world project: Build and deploy an end-to-end deep learning system
- Course wrap-up and next steps in the field

Ready for Day 5! 🚀

**Key Takeaway**: Deep learning is powerful but requires careful data preparation, model selection, and training strategies. Start simple, iterate systematically, and always validate your assumptions!