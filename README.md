# MICRA H SENSE — Emotion Classification from Hormone Biomarkers

## Overview

MICRA H SENSE is a machine learning and rule-based system for emotion state classification using blood hormone biomarkers. The project implements two complementary approaches:

1. **Neural Network Model (MicraSenseNetV2)** — A deep learning classifier using MindSpore framework trained on 200 patient samples with 23 hormone features
2. **Rule-Based Flowchart Engine (V3)** — A clinically-grounded classification system with per-emotion diagnostic algorithms and metabolic confound detection

This project bridges bioinformatics and emotion AI, targeting the Huawei Innovation Track.

## Key Features

- **8-Class Emotion Classification**: Anger, Happiness, Depression, Sadness, Anxiety, Acute Stress, Fear/Panic, Calm
- **23 Biomarkers**: Comprehensive hormone panel including cortisol, catecholamines, monoamines, sex hormones, and inflammatory markers
- **Metabolic Safety Check**: Detects insulin-driven confounds (reactive hypoglycemia masquerading as Fear/Panic)
- **Time-Aware Recovery Logic**: Distinguishes Acute Stress from Anxiety using 90-minute hormone recovery windows
- **Sex-Specific Thresholds**: Accounts for hormonal differences in males vs. females
- **Confidence Grading**: Low-Moderate → Very High confidence assessment per classification
- **Dual Algorithm Support**: Choose between fast neural network inference or interpretable rule-based classification

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Results](#results)
- [References](#references)

## Installation

### Prerequisites
- Python 3.7+
- MindSpore framework
- NumPy, pandas, scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/CorianderEnjoyer97/micrahsense_mindspore_ai.git
cd micrahsense_mindspore_ai

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Neural Network Model (V2)

Train and evaluate the MicraSenseNetV2 deep learning model:

```bash
python micra_h_sense_full_model.py
```

**Output:**
- 400 epochs of training with loss tracking
- Test set accuracy
- Random patient prediction showcase with confidence scores

### 2. Rule-Based Classifier (V3)

Run the clinically-grounded flowchart-based classifier:

```bash
python micra_h_sense_v3.py
```

**Features:**
- Single-reading or multi-reading (time-series) classification
- Detailed marker-by-marker diagnostic breakdown
- Metabolic confound warnings
- Per-emotion confidence levels
- Batch CSV evaluation with per-class accuracy

**Example Usage (Python):**

```python
from micra_h_sense_v3 import classify

# Single reading
result = classify([{
    "timestamp_min": 0,
    "sex": "F",
    "cortisol_nmolL": 720,
    "epinephrine_pgmL": 900,
    "norepinephrine_pgmL": 1300,
    # ... other 20 hormone values
}])

print(f"Emotion: {result.emotion}")
print(f"Confidence: {result.confidence}")
```

## Models

### MicraSenseNetV2 (Neural Network)
- **Architecture**: 4-layer dense network (23 → 128 → 64 → 32 → 16 → 8)
- **Activation**: ReLU hidden layers
- **Optimizer**: Adam (lr=0.005)
- **Loss**: Cross-Entropy
- **Training**: 400 epochs on 160 samples (80/20 split)
- **Normalization**: Min-Max feature scaling

### V3 Flowchart Engine (Rule-Based)
- **8 Independent Flowcharts**: One per emotion state
- **Priority Ordering**: Fear/Panic → Acute Stress → Anger → Anxiety → Depression → Sadness → Happiness → Calm
- **Decision Logic**: 
  - Mandatory marker thresholds
  - Confirmatory evidence scoring
  - Sex-specific rules for females
  - Time-based recovery windows (60–90 min)
  - Confound pre-checks

## Dataset

### micra_h_sense_dataset.csv
- **200 patient records** with 25 columns
- **24 Biomarkers**:
  - Stress hormones: Cortisol, Epinephrine, Norepinephrine, ACTH
  - Monoamines: Serotonin, Dopamine
  - Neuropeptides: Oxytocin, Vasopressin, Prolactin, BDNF
  - Immune: IL-6
  - Neuromodulators: GABA
  - Sex hormones: Testosterone, Estradiol, Progesterone
  - Metabolic: Insulin, Leptin, Melatonin
- **Demographics**: Age, Sex, BMI
- **Labels**: 8-class emotion ground truth

### Data Features
- All hormone values normalized to international units (pg/mL, nmol/L, ng/mL, etc.)
- Balanced class distribution across emotions
- Realistic physiological ranges based on clinical reference intervals

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│          Hormone Biomarker Panel (24 values)            │
├─────────────────────────────────────────────────────────┤
│              Metabolic Confound Check (Insulin)         │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┬──────────────────┬──────────────────┐ │
│  │ Neural (V2)  │ Flowchart (V3)   │  Ensemble Mode   │ │
│  │ ──────────   │ ──────────────   │  ─────────────   │ │
│  │ MindSpore    │ Rule-based 8     │  Vote-based or   │ │
│  │ Deep Net     │ flowcharts +     │  rule+NN fusion  │ │
│  │ 400 epochs   │ sex-specific     │                  │ │
│  │              │ logic            │                  │ │
│  └──────────────┴──────────────────┴──────────────────┘ │
├─────────────────────────────────────────────────────────┤
│   Emotion Classification + Confidence + Diagnostics    │
└─────────────────────────────────────────────────────────┘
```

## Results

### MicraSenseNetV2 Test Accuracy
- **Overall**: ~65–75% (varies with random seed)
- Per-class performance on 40 test samples

### V3 Flowchart Engine
- **Design**: Optimized for interpretability over raw accuracy
- **Strengths**: Clear diagnostic reasoning, metabolic safety, time-awareness
- **Per-Class Markers**: Each emotion has 3+ mandatory thresholds + confirmatory logic

## Project Structure

```
micrahsense_mindspore_ai/
├── README.md                              # This file
├── micra_h_sense_dataset.csv              # 200 patient records
├── micra_h_sense_full_model.py            # Neural network trainer (V2)
├── micra_h_sense_v3.py                    # Rule-based classifier (V3)
├── requirements.txt                       # Dependencies
└── [Jupyter notebooks - future]
```

## References

- **Document**: "Micra H Sense Blood-Based Hormone Algorithm Reference" (Huawei Innovation Track)
- **Framework**: MindSpore (https://www.mindspore.cn/)
- **Approach**: Hybrid neural-symbolic AI combining learned patterns with clinical flowcharts


## Contributors

- CorianderEnjoyer97 (Lead Developer)
- mei1104 (Collaborator)
- peirou0219 (Collaborator)
- rahman93iu-spec (Collaborator)

## License

MIT License — see LICENSE file for details.

---

**Status**: Active development | **Version**: 2.0 (Rule-Based + Neural) | **Last Updated**: 2026-05-22
