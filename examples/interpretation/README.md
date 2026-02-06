# Model Interpretation & Explainability

Comprehensive model interpretation, visualization, and analysis examples.

## Examples (6 total)

- **`interpretation_demo.py`** - Basic GradCAM and interpretation methods
- **`interpretation_metrics_demo.py`** - Quantitative interpretation metrics
- **`interpretation_phase2_demo.py`** - Advanced interpretation techniques
- **`interpretation_phase3_demo.py`** - Training callbacks and monitoring
- **`interactive_visualization_demo.py`** - Interactive Plotly visualizations
- **`comprehensive_interpretation_tutorial.ipynb`** - Complete 40+ cell tutorial

## Quick Start

```bash
# Basic interpretation
python interpretation/interpretation_demo.py

# Interactive visualization
python interpretation/interactive_visualization_demo.py

# Comprehensive tutorial (Jupyter)
jupyter notebook interpretation/comprehensive_interpretation_tutorial.ipynb
```

## Methods Available

- **GradCAM** - Best for CNNs (ResNet, EfficientNet)
- **GradCAM++** - Enhanced localization
- **Integrated Gradients** - Pixel-level attribution
- **Attention Visualization** - For Vision Transformers
- **Guided Backprop** - Fine-grained visualization
- **SmoothGrad** - Noise reduction

## Quantitative Metrics

- **Insertion/Deletion** - Measure explanation quality
- **Sensitivity** - Robustness to perturbations
- **Pointing Game** - Localization accuracy

## Interactive Features

- Zoom, pan, hover on heatmaps
- Side-by-side comparisons
- Method switching
- Export to HTML
