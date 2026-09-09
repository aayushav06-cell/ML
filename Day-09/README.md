# Machine Learning Day 9 Tutorial Notes

After production pipelines and responsible AI, we turn to the generative and visual core of modern deep learning: **advanced CNN architectures, Generative Adversarial Networks (GANs), and Diffusion Models**. These tools power everything from image generation to super-resolution to synthetic data pipelines.

---

## Table of Contents
1. [Why Advanced Deep Learning?](#1-why-advanced-deep-learning)
2. [Modern CNN Architectures](#2-modern-cnn-architectures)
3. [Vision Transformers (A Practical Bridge)](#3-vision-transformers-a-practical-bridge)
4. [Generative Adversarial Networks](#4-generative-adversarial-networks)
5. [Diffusion Models](#5-diffusion-models)
6. [Training Stability and Evaluation](#6-training-stability-and-evaluation)
7. [Hands-On Exercise: DCGAN on CIFAR-10](#7-hands-on-exercise-dcgan-on-cifar-10)
8. [Synthetic Media Safety](#8-synthetic-media-safety)
9. [Common Pitfalls](#9-common-pitfalls)
10. [Summary](#10-summary)
11. [Next Steps](#11-next-steps)

---

## 1. Why Advanced Deep Learning?

Day 5 introduced CNNs as feature extractors. Day 9 goes further: **how modern vision networks are designed, how generative models learn data distributions, and how to ship them responsibly.**

```
Images → CNN/ViT → Features → Classification / Detection / Generation
                                   ↑
Generative models learn P(data) so they can sample new, realistic data
```

Core use cases:
- **Vision**: classification, detection, segmentation, style transfer, super-resolution
- **Synthetic data**: augment scarce datasets, preserve privacy in sensitive domains
- **Generation**: art, design, text-to-image, protein structure visualization
- **Pretraining**: self-supervised vision models downstream of modern CNNs

---

## 2. Modern CNN Architectures

Modern CNNs are no longer "just deeper." They combine **residual connections, factorized convolutions, and attention** to scale efficiently.

### Residual Learning

Plain deep networks saturate; residual networks learn additive residuals:

```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Identity mapping + residual = stable very-deep training."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual          # skip connection
        return self.relu(out)
```

### Factorized and Efficient Convolutions

Inception and MobileNet-style factorizations split computation across small kernels and depthwise separable convs.

```python
class DepthwiseSeparableConv(nn.Module):
    """MobileNet-style factorized convolution: spatial then channel mixing."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.pointwise(self.bn1(self.depthwise(x)))))
```

### Transfer Learning with Modern CNNs

```python
import torchvision.models as models

# Load pretrained backbone, replace head for your dataset
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Freeze early layers, fine-tune the head + later blocks
for name, param in model.named_parameters():
    if not name.startswith("layer4") and not name.startswith("fc"):
        param.requires_grad = False
```

### Architecture Comparison

| Architecture | Key Idea | Best For | Trade-off |
|--------------|----------|----------|-----------|
| **ResNet** | Skip connections | General vision | Baseline |
| **DenseNet** | Concatenate features | Memory-efficient features | High memory |
| **EfficientNet** | Compound scaling | Production accuracy/FLOPs | Requires tuning |
| **MobileNetV3** | Depthwise separable | Edge/mobile | Lower ceiling |
| **ConvNeXt** | Modernized CNN + LN | Competing with ViT | Large kernels |

---

## 3. Vision Transformers (A Practical Bridge)

ViT splits images into patches and applies self-attention. It bridges the Day 6 transformer knowledge into vision.

```python
class PatchEmbedding(nn.Module):
    """Flatten image into patch tokens like a ViT."""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, embed_dim))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)          # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed
```

Use ViT when you have large datasets and strong regularization; use ConvNeXt/MobileNet when compute is constrained or data is modest.

---

## 4. Generative Adversarial Networks

GANs learn a data distribution through a **minimax game** between a generator and a discriminator.

```
Generator (G)                Discriminator (D)
noise → fake image  ──→ D(fake) → fake probability
                           ↑
real image → D(real) → real probability
```

The classic objective:

```python
# Simplified adversarial loss
real_labels = torch.ones(batch_size)
fake_labels = torch.zeros(batch_size)

# Discriminator step
d_real_loss = criterion(D(real_images), real_labels)
d_fake_loss = criterion(D(G(noise)), fake_labels)
d_loss = (d_real_loss + d_fake_loss) / 2

# Generator step
g_loss = criterion(D(G(noise)), real_labels)  # Fool D
```

### DCGAN Architecture Rules

- Use **strided convolutions** instead of pooling in the generator/discriminator.
- Use **BatchNorm** in both networks (except generator output and discriminator input).
- Avoid **fully connected layers** behind large feature maps.
- Use **ReLU** in the generator output layer, **LeakyReLU** in the discriminator.

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)

class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)
```

### Beyond DCGAN: WGAN-GP Intuition

WGAN-GP replaces Jensen-Shannon divergence with the Earth-Mover distance and adds a **gradient penalty** for stable training.

```python
# Gradient penalty for WGAN-GP
def gradient_penalty(D, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
```

---

## 5. Diffusion Models

Diffusion models have become the dominant generative paradigm. They learn to **reverse a gradual noise process** rather than directly model the data distribution.

```
Forward: x₀ → x₁ → … → x_T ≈ N(0, I)     (add noise)
Reverse: x_T → x_{T-1} → … → x₀           (learned denoiser)
```

### Forward Process

At each step, Gaussian noise is added:

```python
import numpy as np

def schedule_beta(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from β_start to β_end."""
    return np.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, noise=None):
    """
    Forward diffusion: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    n = x_start.shape[0]
    betas = schedule_beta(1000)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    sqrt_alpha_bar = torch.sqrt(torch.tensor(alpha_bars[t], dtype=torch.float32))
    sqrt_one_minus = torch.sqrt(torch.tensor(1 - alpha_bars[t], dtype=torch.float32))
    return sqrt_alpha_bar * x_start + sqrt_one_minus * noise
```

### Reverse Process and Loss

The model learns to predict the noise:

```python
# Simplified training objective
def diffusion_loss(model, x_start, t):
    noise = torch.randn_like(x_start)
    x_t = q_sample(x_start, t, noise=noise)
    predicted_noise = model(x_t, t)
    return torch.nn.functional.mse_loss(predicted_noise, noise)
```

### U-Net Backbone

The denoiser is typically a U-Net with skip connections:

```python
class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
```

Sampling inverts the schedule step by step using the learned denoiser. Each step is expensive compared to a GAN, but the quality and stability are far better.

---

## 6. Training Stability and Evaluation

### Stability Tactics

- **Spectral normalization** on discriminator layers.
- **Gradient penalty** for WGAN-GP.
- **Two-time-scale update rule (TTUR)** or separate learning rates.
- **Label smoothing** and occasional no-discriminator updates.
- **EMA of generator weights** for smoother samples.

```python
# Spectral normalization on discriminator convs
for name, module in D.named_modules():
    if isinstance(module, nn.Conv2d):
        setattr(D, name, nn.utils.spectral_norm(module))
```

### Generation Quality Metrics

- **Inception Score (IS)**: confidence and diversity of generated samples.
- **Fréchet Inception Distance (FID)**: distance between real and generated feature distributions.

```python
# Conceptual FID calculation using Inception features
def calculate_fid(real_features, gen_features):
    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(0), np.cov(gen_features, rowvar=False)

    diff = mu_real - mu_gen
    cov_mean, _ = np.sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * cov_mean)
    return fid
```

FID should be computed on a fixed feature extractor and reported with sample counts; never trust a single FID value without confidence intervals.

---

## 7. Hands-On Exercise: DCGAN on CIFAR-10

### Build and Train a DCGAN

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = DCGANGenerator().to(device)
D = DCGANDiscriminator().to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(20):
    for i, (real, _) in enumerate(loader):
        batch = real.to(device)
        labels_real = torch.ones(batch.size(0), device=device)
        labels_fake = torch.zeros(batch.size(0), device=device)

        # Train discriminator
        D.zero_grad()
        loss_real = criterion(D(batch), labels_real)
        noise = torch.randn(batch.size(0), 100, 1, 1, device=device)
        loss_fake = criterion(D(G(noise).detach()), labels_fake)
        loss_d = (loss_real + loss_fake) / 2
        loss_d.backward()
        opt_d.step()

        # Train generator
        G.zero_grad()
        noise = torch.randn(batch.size(0), 100, 1, 1, device=device)
        loss_g = criterion(D(G(noise)), labels_real)
        loss_g.backward()
        opt_g.step()

    print(f"Epoch {epoch}: d_loss={loss_d.item():.4f}, g_loss={loss_g.item():.4f}")

# Sample after training
with torch.no_grad():
    fake = G(torch.randn(16, 100, 1, 1, device=device)).cpu()
```

### Expected Behavior

- `d_loss` starts around 1.3 and settles, `g_loss` decreases steadily.
- Samples should become structured but may still show mode collapse if `g_loss` drops too fast.
- If FID or sample diversity stalls, reduce generator learning rate or add spectral norm.

---

## 8. Synthetic Media Safety

GANs and diffusion models can generate realistic faces, documents, and media. This power carries **ethical obligations** that Day 8 covered, but here are the practical controls:

```python
class SyntheticMediaPolicy:
    """Check before releasing generated content."""
    def __init__(self):
        self.require_consent = True
        self.watermark_generated = True
        self.restrict_face_replication = True

    def audit(self, samples, metadata):
        """Log provenance and constraints."""
        for sample in samples:
            sample["generated"] = True
            sample["model"] = metadata["model_name"]
            sample["checksum"] = hash(sample["bytes"])
        print("✅ Provenance recorded")
        return samples
```

- Always tag generated content.
- Restrict face/voice replication without consent.
- Maintain a usage policy, not just a model card.

---

## 9. Common Pitfalls

### 1. **Confusing GAN Loss Behavior with Quality**
```python
# Low discriminator loss ≠ good samples
# Monitor FID/diversity, not just adversarial losses
```

### 2. **Ignoring Mode Collapse**
```python
# Generator covers only a subset of modes
# Use minibatch discrimination, unrolled GANs, or diversity-promoting losses
```

### 3. **Training GANs with Poor Normalization**
```python
# BatchNorm can fail with small batches
# Use spectral norm + careful learning-rate scheduling
```

### 4. **Diffusion Sampling Without Guidance**
```python
# Classifier-free guidance scales the denoiser toward conditional samples
# Without it, unconditional samples may be blurry and off-distribution
```

### 5. **Skipping Evaluation Beyond Loss**
```python
# GANs and diffusion models need FID/IS/precision-recall, not accuracy
```

### 6. **Releasing Synthetic Media Without Provenance**
```python
# Always watermark and log generation metadata
```

---

## 10. Summary

| Topic | Key Takeaway |
|-------|-------------|
| Modern CNNs | Residual connections, factorized convs, and compound scaling beat plain depth |
| Vision Transformers | ViT bridges transformer knowledge into vision, but compute/data needs differ |
| GANs | Minimax game; DCGAN rules and spectral norm stabilize training |
| Diffusion | Learn to denoise stepwise; higher quality but slower than GANs |
| Stability | Spectral norm, TTUR, gradient penalty, EMA |
| Evaluation | FID/IS, not training loss alone |
| Safety | Provenance, watermarks, and usage policy |

---

## 11. Next Steps

- **Day 10**: Reinforcement Learning, Recommendation Systems, and Multimodal AI
- Implement a WGAN-GP from scratch and compare samples against DCGAN.
- Replace the U-Net above with a small pretrained encoder backbone and measure FID.
- Read: [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) by Salimans et al.
- Read: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by Ho et al.
- Audit a generative pipeline with provenance logging and watermarks.

**Key Takeaway**: Modern deep learning is not just bigger classifiers — it is a design discipline. Residual connections make deep vision stable; GANs and diffusion models learn distributions; evaluation and safety make generation useful rather than merely impressive.

---

*Recommended tools for practice*: PyTorch, torchvision, torchaudio/torchserve for export, `torchsummary` for profiling, `scipy` for FID, and IBM AI Fairness 360 when synthetic data carries bias from Day 8.
