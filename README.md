# ARQUE: Anisotropic Richness Quality Estimation
### A Hybrid Multi-Expert Framework for No-Reference Image Quality Assessment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Paper_Submitted-orange)

**ARQUE** is a novel No-Reference Image Quality Assessment (NR-IQA) framework that adopts a **"Divide and Conquer"** strategy. Instead of using a single generalist model, ARQUE employs a probabilistic classifier to route images to specialized Support Vector Regressors (SVRs), each optimized for specific physical distortions (Blur, Noise, Compression, etc.).

This repository contains the official implementation and reproduction scripts for the paper:  
> **"ARQUE: A Hybrid Multi-Expert Framework for No-Reference Image Quality Assessment Using Curvature Analysis"**

---

## 🚀 Key Features

* **State-of-the-Art Performance:** Achieves **PLCC 0.954** on the LIVE dataset, outperforming BRISQUE (0.874) and NIQE (0.915).
* **Physically Interpretable:** Based on the **Anisotropic Texture Richness (ATR)** metric, which measures structural integrity via bi-directional curvature analysis.
* **High Efficiency:** The specialist architecture reduces regression complexity by **72.5%** compared to generalist baselines (fewer active support vectors required).
* **Robust Generalization:** Validated on both LIVE and CSIQ datasets.

---

## 📊 Benchmark Results

### 1. Performance on LIVE Dataset (Release 2)
Comparison against standard baselines (re-implemented and optimized under identical conditions).

| Method | Type | PLCC (Linearity) | SROCC (Rank) | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **BRISQUE** (Optimized) | Generalist | 0.8744 | 0.8831 | 8.80 |
| **NIQE** | Unaware | 0.9150 | 0.9130 | - |
| **ARQUE (Ours)** | **Multi-Expert** | **0.9541** | **0.9551** | **6.47** |

### 2. Generalization on CSIQ Dataset
Results using the auto-calibration module (intra-dataset test).

| Method | PLCC | RMSE |
| :--- | :--- | :--- |
| **BRISQUE** (Baseline) | 0.707 | 210.96 |
| **ARQUE (Ours)** | **0.804** | **173.04** |

---

## 📂 Repository Structure

```text
ARQUE-IQA/
├── data/                  # Dataset placeholder (See README.txt inside)
├── models/                # Pre-trained models (.pkl) and configs (.json)
│   ├── classifier_hybrid.pkl
│   ├── svr_specialists_pro.pkl
│   └── trained_models_LIVE.json
├── scripts/               # Reproduction scripts
│   ├── 1_reproduce_live.py
│   ├── 2_reproduce_csiq.py
│   └── 3_check_complexity.py
├── src/                   # Core logic (ATR extraction, NSS, System Class)
│   ├── __init__.py
│   └── arque_core.py
├── requirements.txt       # Python dependencies
└── README.md              # This file
