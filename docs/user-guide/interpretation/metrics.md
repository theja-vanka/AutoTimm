# Explanation Quality Metrics

Quantitatively evaluate the quality of your interpretations using standardized metrics.

## Overview

While visual inspection of heatmaps is helpful, quantitative metrics provide objective measures of explanation quality:

- **Faithfulness**: Does the explanation reflect model behavior?
- **Sensitivity**: Is the explanation stable under perturbations?
- **Sanity Checks**: Does the method pass basic reasonableness tests?
- **Localization**: Does the explanation correctly identify important regions?

## Class: `ExplanationMetrics`

```python
from autotimm.interpretation import ExplanationMetrics, GradCAM

model = ImageClassifier(backbone="resnet50", num_classes=10)
explainer = GradCAM(model)
metrics = ExplanationMetrics(model, explainer)
```

---

## Faithfulness Metrics

Faithfulness metrics measure how well the explanation reflects the model's actual behavior.

### Deletion

**Concept**: Progressively remove the most important pixels (according to the explanation) and measure how much the prediction drops.

**Expected behavior**: If the explanation is faithful, removing important pixels should significantly decrease confidence.

```python
result = metrics.deletion(
    image,
    target_class=None,  # None = predicted class
    steps=50,           # Number of deletion steps
    baseline='blur'     # What to replace deleted pixels with
)

print(f"Deletion AUC: {result['auc']:.3f}")
print(f"Final drop: {result['final_drop']:.2%}")
```

**Returned fields:**
- `auc`: Area under the curve (lower = better)
- `final_drop`: Final prediction drop (higher = better)
- `scores`: List of prediction scores at each step
- `original_score`: Original prediction score

**Interpretation:**
- **AUC < 0.7**: Excellent - important pixels have strong impact
- **AUC 0.7-0.9**: Good - reasonable impact
- **AUC > 0.9**: Poor - removing "important" pixels doesn't affect prediction much

**Baseline options:**
- `'black'`: Replace with zeros
- `'blur'`: Replace with Gaussian blur (recommended)
- `'mean'`: Replace with mean pixel value

### Insertion

**Concept**: Start with a baseline (e.g., blurred) image and progressively add back the most important pixels.

**Expected behavior**: Adding important pixels should quickly recover the original prediction.

```python
result = metrics.insertion(
    image,
    target_class=None,
    steps=50,
    baseline='blur'
)

print(f"Insertion AUC: {result['auc']:.3f}")
print(f"Final rise: {result['final_rise']:.2%}")
```

**Returned fields:**
- `auc`: Area under the curve (higher = better)
- `final_rise`: How much prediction recovers (higher = better)
- `scores`: List of prediction scores at each step
- `baseline_score`: Score on baseline image
- `original_score`: Original prediction score

**Interpretation:**
- **AUC > 0.7**: Excellent - important pixels quickly recover prediction
- **AUC 0.5-0.7**: Good
- **AUC < 0.5**: Poor - "important" pixels don't help prediction

### Visualizing Curves

```python
import matplotlib.pyplot as plt

deletion_result = metrics.deletion(image, steps=50)
insertion_result = metrics.insertion(image, steps=50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Deletion curve
ax1.plot(deletion_result['scores'])
ax1.set_title('Deletion Curve')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Prediction Score')

# Insertion curve
ax2.plot(insertion_result['scores'])
ax2.set_title('Insertion Curve')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Prediction Score')

plt.savefig('faithfulness_curves.png')
```

---

## Sensitivity Analysis

Measures explanation stability under input perturbations.

### Sensitivity-N

**Concept**: Add random noise to the input and measure how much the explanation changes.

**Expected behavior**: Small input changes should lead to small explanation changes.

```python
result = metrics.sensitivity_n(
    image,
    target_class=None,
    n_samples=50,      # Number of noisy samples
    noise_level=0.15   # Std of Gaussian noise
)

print(f"Sensitivity: {result['sensitivity']:.4f}")
print(f"Std: {result['std']:.4f}")
```

**Returned fields:**
- `sensitivity`: Average explanation change
- `std`: Standard deviation of changes
- `max_change`: Maximum change observed
- `changes`: List of all changes

**Interpretation:**
- **< 0.05**: Very stable (excellent)
- **0.05-0.15**: Moderately stable (good)
- **> 0.15**: Unstable (poor)

**Adjusting parameters:**
- Higher `n_samples`: More accurate but slower (50 recommended)
- Higher `noise_level`: Tests stability under larger perturbations

---

## Sanity Checks

Tests whether the explanation method behaves reasonably.

### Model Parameter Randomization

**Concept**: Explanation should change significantly if model weights are randomized.

**Rationale**: If explanations don't change when the model is randomized, the method isn't actually explaining the model's behavior.

```python
result = metrics.model_parameter_randomization_test(image)

print(f"Correlation: {result['correlation']:.3f}")
print(f"Passes: {result['passes']}")
```

**Returned fields:**
- `correlation`: Correlation between original and randomized explanations
- `change`: Mean absolute difference
- `passes`: True if correlation < 0.5

**Interpretation:**
- **passes=True**: Method is model-sensitive ✓
- **passes=False**: ⚠ Method may not be explaining the model

### Data Randomization

**Concept**: Explanation should change significantly when explaining a different class.

**Rationale**: Explanations for different classes should look different.

```python
result = metrics.data_randomization_test(image)

print(f"Correlation: {result['correlation']:.3f}")
print(f"Passes: {result['passes']}")
```

**Returned fields:**
- `correlation`: Correlation with explanation for random class
- `change`: Mean absolute difference
- `passes`: True if correlation < 0.5

**Interpretation:**
- **passes=True**: Method is class-sensitive ✓
- **passes=False**: ⚠ Method produces similar explanations for all classes

---

## Localization Metrics

For tasks where object location is known (e.g., object detection with annotations).

### Pointing Game

**Concept**: Does the maximum attention fall within the object's bounding box?

```python
bbox = [x1, y1, x2, y2]  # Ground truth bounding box

result = metrics.pointing_game(image, bbox)

print(f"Hit: {result['hit']}")
print(f"Max location: {result['max_location']}")
```

**Returned fields:**
- `hit`: True if max attention is inside bbox
- `max_location`: (y, x) coordinates of maximum attention
- `bbox`: Input bbox

**Interpretation:**
- **hit=True**: Explanation correctly localizes object ✓
- **hit=False**: Explanation misses object

**Use cases:**
- Evaluate detection explanations
- Validate that attention focuses on relevant objects
- Compare localization accuracy across methods

---

## Comprehensive Evaluation

Run all applicable metrics at once:

```python
results = metrics.evaluate_all(
    image,
    target_class=None,
    bbox=None  # Optional: for pointing game
)

# Access individual metric results
print(f"Deletion AUC: {results['deletion']['auc']:.3f}")
print(f"Insertion AUC: {results['insertion']['auc']:.3f}")
print(f"Sensitivity: {results['sensitivity']['sensitivity']:.4f}")
print(f"Param check: {results['param_randomization']['passes']}")
print(f"Data check: {results['data_randomization']['passes']}")

# If bbox provided
if 'pointing_game' in results:
    print(f"Pointing game: {results['pointing_game']['hit']}")
```

---

## Comparing Methods

Evaluate multiple explanation methods:

```python
from autotimm.interpretation import GradCAM, GradCAMPlusPlus, IntegratedGradients

methods = {
    'GradCAM': GradCAM(model),
    'GradCAM++': GradCAMPlusPlus(model),
    'IntegratedGradients': IntegratedGradients(model),
}

print(f"{'Method':<20} {'Del AUC':<10} {'Ins AUC':<10} {'Sensitivity':<12}")
print("-" * 52)

for name, explainer in methods.items():
    metrics_obj = ExplanationMetrics(model, explainer, use_cuda=False)

    deletion = metrics_obj.deletion(image, steps=20)
    insertion = metrics_obj.insertion(image, steps=20)
    sensitivity = metrics_obj.sensitivity_n(image, n_samples=20)

    print(f"{name:<20} {deletion['auc']:<10.3f} {insertion['auc']:<10.3f} "
          f"{sensitivity['sensitivity']:<12.4f}")
```

**Output example:**
```
Method               Del AUC    Ins AUC    Sensitivity
----------------------------------------------------
GradCAM              0.723      0.812      0.0441
GradCAM++            0.698      0.835      0.0816
IntegratedGradients  0.654      0.891      0.0569
```

**Analysis:**
- IntegratedGradients has best faithfulness (lowest deletion, highest insertion)
- GradCAM has best stability (lowest sensitivity)
- Trade-off between faithfulness and efficiency

---

## Best Practices

### 1. Use Multiple Metrics

Don't rely on a single metric. Good explanations should:
- Have low deletion AUC (<0.8)
- Have high insertion AUC (>0.6)
- Have low sensitivity (<0.15)
- Pass both sanity checks

### 2. Consider Your Use Case

**For model debugging:**
- Focus on faithfulness (deletion/insertion)
- Sanity checks are critical

**For production deployment:**
- Prioritize stability (sensitivity)
- Balance with faithfulness

**For scientific research:**
- Report all metrics
- Include significance tests

### 3. Appropriate Baselines

```python
# For natural images - use blur
result = metrics.deletion(image, baseline='blur')

# For medical images - consider mean
result = metrics.deletion(image, baseline='mean')

# For stylized/synthetic - black may work
result = metrics.deletion(image, baseline='black')
```

### 4. Sufficient Steps

```python
# Too few steps (10) - coarse measurement
result = metrics.deletion(image, steps=10)

# Good balance (50) - recommended
result = metrics.deletion(image, steps=50)

# Many steps (100) - more accurate but slower
result = metrics.deletion(image, steps=100)
```

---

## Performance Considerations

### Computational Cost

| Metric | Relative Cost | Notes |
|--------|---------------|-------|
| Deletion | High | steps × forward passes |
| Insertion | High | steps × forward passes |
| Sensitivity | High | n_samples × forward passes |
| Param randomization | Medium | 2 × forward passes + weight copy |
| Data randomization | Low | 2 × forward passes |
| Pointing game | Low | 1 × forward pass |

### Optimization Tips

```python
# Use fewer steps for quick evaluation
quick_result = metrics.deletion(image, steps=10)

# Use fewer samples for sensitivity
quick_sensitivity = metrics.sensitivity_n(image, n_samples=10)

# Skip expensive metrics for large-scale evaluation
# (only run on a sample of images)
```

### Batching (Future Enhancement)

Currently, metrics are computed per-image. For evaluation on datasets, process in batches manually:

```python
all_deletions = []
for image in dataset:
    result = metrics.deletion(image, steps=20)
    all_deletions.append(result['auc'])

mean_deletion = np.mean(all_deletions)
print(f"Mean deletion AUC: {mean_deletion:.3f}")
```

---

## Troubleshooting

### High Deletion AUC (>0.9)

**Problem**: Explanation doesn't affect prediction much

**Solutions:**
- Check that explanation method is working (visualize heatmap)
- Try different target layer
- Verify model is trained
- Consider different baseline

### Sensitivity NaN

**Problem**: `sensitivity` is NaN

**Cause**: Heatmaps are constant (no variation)

**Solutions:**
- Check that model produces varied predictions
- Verify explanation method is working
- Ensure input isn't constant

### Sanity Checks Fail

**Problem**: `passes=False` for sanity checks

**Investigation:**
- Visualize explanations for different classes
- Check correlation value (how close to threshold)
- Some simple models may legitimately have high correlation

### Pointing Game Always Misses

**Problem**: `hit=False` consistently

**Solutions:**
- Verify bbox coordinates are correct
- Check heatmap size matches image size
- Try different explanation method
- Visualize max attention location

---

## Examples

See `examples/interpretation_metrics_demo.py` for comprehensive examples including:
- Computing faithfulness metrics
- Analyzing sensitivity
- Running sanity checks
- Comparing multiple methods
- Creating visualizations

---

## References

**Deletion & Insertion:**
- Petsiuk et al., "RISE: Randomized Input Sampling for Explanation of Black-box Models", BMVC 2018

**Sensitivity:**
- Yeh et al., "On the (In)fidelity and Sensitivity of Explanations", NeurIPS 2019

**Sanity Checks:**
- Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018

**Pointing Game:**
- Zhang et al., "Top-down Neural Attention by Excitation Backprop", IJCV 2018

---

## See Also

- [Interpretation Methods](methods.md) - Available explanation methods
- [Main Guide](index.md) - Overview and quick start
- [Feature Visualization](feature-visualization.md) - Analyze learned features
