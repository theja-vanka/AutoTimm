# Interpretation Issues

Problems with model interpretation and explanation methods.

## No Heatmap Visible

**Solutions:**

```python
# 1. Check target layer is correct
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(name)

# 2. Verify model is in eval mode
model.eval()

# 3. Ensure gradients are enabled
```

## Poor Explanation Localization

**Solutions:**

```python
# 1. Try GradCAM++ instead of GradCAM
from autotimm.interpretation import GradCAMPlusPlus

explainer = GradCAMPlusPlus(model, target_layer="backbone.layer4")

# 2. Use different target layer
explainer = GradCAM(model, target_layer="backbone.layer3")

# 3. Apply SmoothGrad for cleaner results
```

## Detection Explanations: No Explanations Generated

**Solutions:**

```python
# 1. Lower detection threshold
from autotimm.interpretation import explain_detection

results = explain_detection(
    model,
    image,
    detection_threshold=0.1,  # Lower threshold
)

# 2. Debug raw detections
print(f"Number of detections: {len(results['detections'])}")
```

## Segmentation Explanations: Blank Heatmaps

**Solutions:**

```python
# 1. Check if target class exists
from autotimm.interpretation import explain_segmentation

results = explain_segmentation(model, image)
unique_classes = np.unique(results['prediction'])
print(f"Present classes: {unique_classes}")

# 2. Only explain present classes
for class_id in unique_classes:
    explain_segmentation(
        model,
        image,
        target_class=int(class_id),
        save_path=f'class_{class_id}.png'
    )
```

## Interpretation Metrics: High Deletion AUC

**Problem:** Explanation doesn't affect prediction (>0.9)

**Solutions:**

```python
# 1. Visualize heatmap to verify it's working
# 2. Try different target layer
# 3. Verify model is trained (not random)
# 4. Consider different baseline
```

## Interpretation Callbacks: Not Logging

**Solutions:**

```python
# 1. Verify logger is configured
# 2. Check log_every_n_epochs setting
# 3. Ensure training runs long enough

# Debug: Log every epoch
callback = InterpretationCallback(
    sample_images=images,
    log_every_n_epochs=1,  # Log every epoch
)
```

## Feature Visualization: ValueError Layer Not Found

**Solution:** Check available layers:

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(name)
```

## Related Issues

- [Model Loading](../models/model-loading.md) - Model compatibility
- [Loggers](../integration/loggers.md) - Logging interpretation results
