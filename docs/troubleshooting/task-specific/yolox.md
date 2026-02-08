# YOLOX Training Issues

Problems specific to YOLOX object detection.

## CUDA Out of Memory with YOLOX

**Solutions:**

```python
# 1. Reduce batch size
data = DetectionDataModule(batch_size=8, ...)

# 2. Use gradient accumulation for effective larger batch
trainer = AutoTrainer(accumulate_grad_batches=8, ...)

# 3. Use smaller model
model = YOLOXDetector(model_name="yolox-s", ...)  # Instead of yolox-l
```

## YOLOX Slow Training

**Solutions:**

```python
# 1. Use mixed precision
trainer = AutoTrainer(precision="16-mixed", ...)

# 2. Reduce image size
data = DetectionDataModule(image_size=416, ...)

# 3. Reduce workers if CPU-bound
data = DetectionDataModule(num_workers=2, ...)
```

## YOLOX Poor Performance

**Solution:** Use official settings:

```python
model = YOLOXDetector(
    model_name="yolox-s",
    lr=0.01,              # Official LR
    optimizer="sgd",      # SGD, not AdamW
    scheduler="yolox",    # YOLOX scheduler
    total_epochs=300,     # Full training
)

data = DetectionDataModule(batch_size=64, ...)  # Proper batch size
```

## Related Issues

- [OOM Errors](../performance/oom-errors.md) - Memory optimization
- [Slow Training](../performance/slow-training.md) - Performance tips
- [Convergence](../training/convergence.md) - Training issues
