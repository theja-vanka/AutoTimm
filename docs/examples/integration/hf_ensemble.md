# Model Ensemble & Knowledge Distillation

Combine multiple models and distill knowledge into compact students for production deployment.

## Ensemble & Distillation Architecture

```mermaid
graph TD
    A[Multiple Models] --> A1[Model 1]
    A --> A2[Model 2]
    A --> A3[Model N]
    
    A1 --> A1a[Forward Pass]
    A2 --> A2a[Forward Pass]
    A3 --> A3a[Forward Pass]
    
    A1a --> B{Combination}
    A2a --> B
    A3a --> B
    
    B -->|Average| C1[Simple Average]
    C1 --> C1a[Sum Predictions]
    C1a --> C1b[Divide by N]
    
    B -->|Weighted| C2[Weighted Average]
    C2 --> C2a[Learn Weights]
    C2a --> C2b[Weighted Sum]
    
    C1b --> C[Ensemble Prediction]
    C2b --> C
    
    C --> C3[Aggregate Logits]
    C3 --> C4[Final Prediction]
    C4 -."Soft Targets".-> D[Teacher Model]
    
    D --> D1[Generate Soft Labels]
    D1 --> D2[Temperature Scaling]
    D2 --> D3[Soft Distributions]
    D3 --> E[Soft Targets + Hard Loss]
    
    E --> E1[KL Divergence]
    E1 --> E2[Cross Entropy]
    E2 --> E3[Combined Loss]
    E3 --> F[Student Model]
    
    F --> F1[Smaller Architecture]
    F1 --> F2[Train with Teacher]
    F2 --> F3[Match Distributions]
    F3 --> F4[Optimize Loss]
    F4 --> G[Compact Model]
    
    G --> G1[Reduced Parameters]
    G1 --> G2[Faster Inference]
    G2 --> G3[Production Ready]
    G3 --> G4[Deploy]

    style A fill:#2196F3,stroke:#1976D2
    style B fill:#1976D2,stroke:#1565C0
    style C fill:#2196F3,stroke:#1976D2
    style D fill:#1976D2,stroke:#1565C0
    style F fill:#2196F3,stroke:#1976D2
    style G fill:#1976D2,stroke:#1565C0
```

## Overview

Learn how to create model ensembles for improved accuracy and use knowledge distillation to compress large models into efficient students while retaining performance.

## What This Example Covers

- **Simple averaging ensemble** - Quick accuracy boost
- **Weighted ensemble** - Learned combination weights
- **Knowledge distillation** - Teacher-student training
- **Multi-objective trade-offs** - Accuracy vs speed
- **Production deployment** - Choosing the right approach

## Ensemble Methods

### Simple Averaging

```python
from autotimm import ImageClassifier

# Create diverse models
models = [
    ImageClassifier(backbone="hf-hub:timm/resnet18.a1_in1k", num_classes=10),
    ImageClassifier(backbone="hf-hub:timm/mobilenetv3_small_100.lamb_in1k", num_classes=10),
    ImageClassifier(backbone="hf-hub:timm/efficientnet_b0.ra_in1k", num_classes=10),
]

# Average predictions
outputs = [model(image) for model in models]
ensemble_output = torch.stack(outputs).mean(dim=0)
```

**Benefits**:
- 2-3% accuracy improvement
- No training required
- Works best with diverse architectures

### Weighted Ensemble

```python
class WeightedEnsemble(nn.Module):
    def __init__(self, models, num_classes):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def forward(self, x):
        predictions = [model(x) for model in self.models]
        normalized_weights = F.softmax(self.weights, dim=0)
        return sum(w * pred for w, pred in zip(normalized_weights, predictions))
```

**Benefits**:
- 2.5-4% accuracy improvement
- Learns optimal combination
- Minimal training (weights only)

## Knowledge Distillation

### Basic Distillation

```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        # Hard target loss
        hard_loss = F.cross_entropy(student_logits, labels)

        # Soft target loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss *= (self.temperature ** 2)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# Training
teacher = ImageClassifier(backbone="hf-hub:timm/resnet50.a1_in1k", num_classes=10)
student = ImageClassifier(backbone="hf-hub:timm/mobilenetv3_small_100.lamb_in1k", num_classes=10)

distill_loss = DistillationLoss(alpha=0.7, temperature=3.0)

for images, labels in train_loader:
    with torch.inference_mode():
        teacher_logits = teacher(images)

    student_logits = student(images)
    loss = distill_loss(student_logits, teacher_logits, labels)
```

## Run the Example

```bash
python examples/huggingface/hf_ensemble.py
```

## Comparison Table

| Method | Accuracy Gain | Inference Time | Memory | Training Cost |
|--------|---------------|----------------|---------|---------------|
| Simple Ensemble | +2-3% | N × single | N × single | Low |
| Weighted Ensemble | +2.5-4% | N × single | N × single | Low |
| Knowledge Distillation | +1-2% | 1 × student | 1 × student | High |

## Decision Guide

### Use Simple Ensemble When:
- Inference time is not critical
- You have multiple trained models
- Want quick accuracy boost
- Offline/batch processing

### Use Weighted Ensemble When:
- Have validation set to optimize weights
- Models have varying quality
- Can afford weight optimization
- Need better than simple averaging

### Use Knowledge Distillation When:
- Need fast inference (production/edge)
- Have limited memory budget
- Can afford training time
- Have strong teacher model

### Hybrid Approach:
1. Train ensemble as teacher
2. Distill into single student
3. Best of both worlds!

## Hyperparameters

### Temperature (T)
- **T=1**: No distillation (hard targets only)
- **T=3-5**: Typical range for vision
- **T=10+**: Very soft, more regularization
- Higher T for larger teacher-student gap

### Alpha (α)
- **α=0.5**: Equal weight to hard and soft
- **α=0.7-0.9**: Emphasize soft targets (typical)
- **α=0.1-0.3**: Emphasize hard targets
- Higher α when teacher is very strong

### Student Size
- **0.1-0.3x teacher**: Typical compression
- Too small → Limited capacity
- Too large → Defeats purpose

## Typical Results

```
Teacher (ResNet-50):        91% accuracy
Student from scratch:       85% accuracy
Student with distillation:  88% accuracy  (+3%)
Compression ratio:          10x smaller, 10x faster
```

## Best Practices

1. **Ensemble diversity**: Use different architectures (ResNet + EfficientNet + ViT)
2. **Temperature tuning**: Start with T=3, adjust based on results
3. **Student capacity**: 10-30% of teacher size works well
4. **Training duration**: Train student 2-3x longer than normal
5. **Monitor metrics**: Track both accuracy and inference speed

## Related Examples

- [HuggingFace Hub Models](huggingface-hub.md)
- [Model Deployment](hf_deployment.md)
- [Hyperparameter Tuning](../utilities/hf_hyperparameter_tuning.md)
