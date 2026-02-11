# Interactive Visualizations

Interactive visualizations provide Plotly-based HTML reports for exploring model interpretations in web browsers with zoom, pan, and hover capabilities.

## Installation

Interactive visualizations require Plotly:

```bash
# Install plotly separately
pip install plotly

# Or install with autotimm
pip install autotimm[interactive]

# Or install all optional dependencies
pip install autotimm[all]
```

## Overview

The `InteractiveVisualizer` class provides three main visualization types:

1. **Single Method Visualization** - Interactive heatmap overlay
2. **Method Comparison** - Side-by-side comparison of multiple methods
3. **HTML Reports** - Comprehensive reports with statistics

## Basic Usage

### Single Explanation

```python
from autotimm import ImageClassifier
from autotimm.interpretation import GradCAM, InteractiveVisualizer

model = ImageClassifier(backbone="resnet50", num_classes=10)
explainer = GradCAM(model)
viz = InteractiveVisualizer(model)

# Create interactive visualization
fig = viz.visualize_explanation(
    image,
    explainer,
    target_class=5,
    title="GradCAM Explanation",
    colorscale="Viridis",
    opacity=0.6,
    save_path="explanation.html"
)

# Display in browser or notebook
fig.show()
```

### Method Comparison

Compare multiple interpretation methods side-by-side:

```python
from autotimm.interpretation import GradCAM, GradCAMPlusPlus, IntegratedGradients

explainers = {
    'GradCAM': GradCAM(model),
    'GradCAM++': GradCAMPlusPlus(model),
    'Integrated Gradients': IntegratedGradients(model),
}

fig = viz.compare_methods(
    image,
    explainers,
    target_class=5,
    title="Method Comparison",
    save_path="comparison.html",
    width=1400,
    height=500
)
```

### Comprehensive Report

Generate a complete HTML report with statistics and visualizations:

```python
report_path = viz.create_report(
    image,
    explainer,
    target_class=5,
    include_statistics=True,
    save_path="report.html",
    title="Model Interpretation Report"
)
```

The report includes:
- Prediction information and top-5 classes
- Heatmap statistics (mean, std, min, max, sparsity)
- Interactive visualization
- Importance distribution histogram

## Features

### Interactive Controls

**Zoom**: Mouse wheel or pinch gesture
**Pan**: Click and drag
**Hover**: See exact importance values
**Reset**: Double-click to reset view

### Customization Options

```python
viz.visualize_explanation(
    image,
    explainer,
    target_class=None,          # Target class (None = predicted)
    title="Explanation",         # Figure title
    colorscale="Viridis",       # Color scheme
    opacity=0.6,                # Overlay transparency (0.0-1.0)
    save_path="output.html",    # Save location
    show_colorbar=True,         # Show color legend
    width=800,                  # Figure width
    height=600                  # Figure height
)
```

### Available Colorscales

**Perceptually Uniform** (recommended):
- `Viridis` (default)
- `Plasma`
- `Inferno`
- `Cividis`

**Traditional**:
- `Hot`
- `Jet`
- `Rainbow`

**Diverging**:
- `RdBu` (Red-Blue)
- `RdYlGn` (Red-Yellow-Green)

**Grayscale**:
- `Greys`
- `Blackbody`

## Output Format

Generated HTML files are:
- **Standalone** - No external dependencies except Plotly.js CDN
- **Shareable** - Just send the HTML file
- **Embeddable** - Can be embedded in dashboards or web pages
- **Portable** - Work in any modern browser

Typical file size: 2-5 MB per visualization

## Browser Compatibility

**Desktop**:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Mobile**:
- iOS Safari 14+
- Chrome Mobile 90+

## Use Cases

### Research & Development

```python
# Explore different methods
explainers = {
    'GradCAM': GradCAM(model),
    'GradCAM++': GradCAMPlusPlus(model),
    'Integrated Gradients': IntegratedGradients(model),
}

fig = viz.compare_methods(image, explainers, save_path="research.html")
# Share HTML file with team
```

### Model Debugging

```python
# Generate reports for failing cases
for test_image in failing_cases:
    viz.create_report(
        test_image,
        explainer,
        save_path=f"debug_{test_image.id}.html"
    )
```

### Production Monitoring

```python
# Generate explanations for low-confidence predictions
if confidence < 0.7:
    viz.visualize_explanation(
        image,
        explainer,
        save_path=f"monitoring/{timestamp}.html"
    )
```

### User-Facing Applications

```python
# Provide explanations to end users
@app.route('/explain/<image_id>')
def explain_prediction(image_id):
    image = load_image(image_id)
    report_path = viz.create_report(
        image,
        explainer,
        save_path=f"temp/{image_id}.html"
    )
    return send_file(report_path)
```

## Performance

**Generation Time**:
- Single visualization: ~1-2 seconds
- Method comparison (3 methods): ~3-5 seconds
- Comprehensive report: ~2-3 seconds

**File Sizes**:
- Basic visualization: ~2 MB
- Method comparison: ~4-6 MB
- Comprehensive report: ~3-4 MB

**Browser Performance**:
- Fast loading (<1 second)
- Smooth zooming and panning
- Low memory usage
- Mobile device compatible

## Comparison with Static Visualizations

| Feature | Static (PNG/JPG) | Interactive (HTML) |
|---------|------------------|-------------------|
| **Zoom** | :material-close-circle: No | :material-check-circle: Yes |
| **Pan** | :material-close-circle: No | :material-check-circle: Yes |
| **Hover Info** | :material-close-circle: No | :material-check-circle: Yes |
| **Resolution** | Fixed | Scalable |
| **File Size** | Small (~100KB) | Larger (~2-5MB) |
| **Sharing** | Easy | Easy (single HTML) |
| **Exploration** | Limited | Extensive |
| **Tools Needed** | Any viewer | Just browser |

**Recommendation**:
- Use **interactive** for detailed analysis, presentations, user-facing applications
- Use **static** for quick checks, publications (print), when file size matters

## Graceful Degradation

If Plotly is not installed:

```python
from autotimm.interpretation import InteractiveVisualizer

if InteractiveVisualizer is None:
    print("Plotly not installed. Install with: pip install plotly")
else:
    viz = InteractiveVisualizer(model)
```

## Best Practices

1. **Choose appropriate colorscale**: Use perceptually uniform colormaps (Viridis, Plasma) for scientific accuracy
2. **Adjust opacity**: 0.5-0.7 works well for most cases
3. **Save for sharing**: Generate HTML files for team collaboration
4. **Monitor file sizes**: Use lower resolution images if file size is a concern
5. **Test in target browser**: Verify compatibility with your target audience's browsers

## API Reference

### InteractiveVisualizer

```python
viz = InteractiveVisualizer(model)
```

#### Methods

**visualize_explanation()**
```python
fig = viz.visualize_explanation(
    image,
    explainer,
    target_class=None,
    title="Model Explanation",
    colorscale="Viridis",
    opacity=0.6,
    save_path=None,
    show_colorbar=True,
    width=800,
    height=600
)
```

**compare_methods()**
```python
fig = viz.compare_methods(
    image,
    explainers: Dict[str, object],
    target_class=None,
    title="Method Comparison",
    colorscale="Viridis",
    opacity=0.6,
    save_path=None,
    width=1200,
    height=400
)
```

**create_report()**
```python
report_path = viz.create_report(
    image,
    explainer,
    target_class=None,
    include_statistics=True,
    include_top_regions=True,
    top_k=5,
    save_path="report.html",
    title="Interpretation Report"
)
```

## Examples

See the complete example script:
```
examples/interpretation/interactive_visualization_demo.py
```

See the tutorial notebook:
```
examples/interpretation/comprehensive_interpretation_tutorial.ipynb
```

## Next Steps

- Learn about [Performance Optimization](optimization.md)
- Explore [Quality Metrics](metrics.md)
- See [Interpretation Methods](methods.md)
