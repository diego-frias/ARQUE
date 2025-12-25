
```markdown
# ARQUE: Anisotropic Richness Quality Estimation
### A Hybrid Multi-Expert Framework for No-Reference Image Quality Assessment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Paper_Submitted-orange)

**ARQUE** is a novel No-Reference Image Quality Assessment (NR-IQA) framework that adopts a **"Distortion-Specific Decomposition"** strategy. Instead of using a single generalist model, ARQUE employs a probabilistic classifier to route images to specialized Support Vector Regressors (SVRs), using a **Soft-Voting** mechanism to fuse expert predictions.

This repository contains the official implementation and reproduction scripts for the paper:  
> **"ARQUE: A Hybrid Multi-Expert Framework for No-Reference Image Quality Assessment Using Curvature Analysis"**
```
---

## 🚀 Key Features

* **State-of-the-Art Performance:** Achieves **PLCC 0.947** on the LIVE dataset, outperforming a highly optimized BRISQUE baseline (0.907) and NIQE (0.915).
* **Physically Interpretable:** Based on the **Anisotropic Texture Richness (ATR)** metric, which measures structural integrity via bi-directional curvature analysis.
* **High Efficiency:** The specialist architecture reduces regression complexity by **72.5%** compared to generalist baselines (fewer active support vectors required).
* **Robust Generalization:** Validated on both LIVE and CSIQ datasets using auto-calibration.

---

## 📊 Benchmark Results

### 1. Performance on LIVE Dataset (Release 2)
Comparison against standard baselines. Note that our BRISQUE baseline was **re-implemented and fully optimized** (Grid Search) to ensure a fair and rigorous comparison.

| Method | Type | PLCC (Linearity) | SROCC (Rank) | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **BRISQUE** (Optimized) | Generalist | 0.9065 | 0.9114 | 9.00 |
| **NIQE** | Unaware | 0.9150 | 0.9130 | - |
| **ARQUE (Ours)** | **Multi-Expert** | **0.9465** | **0.9510** | **6.89** |

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
│   ├── 1_reproduce_live.py         # Validates Table 1 (ARQUE performance)
│   ├── 2_reproduce_csiq.py         # Validates Table 2 (Generalization)
│   ├── 3_check_complexity.py       # Audits SVR model size/complexity
│   └── 4_benchmark_vs_brisque.py   # Full comparative benchmark (ARQUE vs BRISQUE)
├── src/                   # Core logic (ATR extraction, NSS, System Class)
│   ├── __init__.py
│   └── arque_core.py      # Main Class with Soft-Voting logic
├── requirements.txt       # Python dependencies
└── README.md              # This file

```

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/YourUsername/ARQUE-IQA.git](https://github.com/YourUsername/ARQUE-IQA.git)
cd ARQUE-IQA

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Setup Datasets:**
Download LIVE (Release 2) and CSIQ databases from their official websites.
Place them in the `data/` folder following the structure described in `data/README.txt`.

---

## 🔄 Reproducibility

We provide four scripts to verify the paper's claims:

**Experiment 1: LIVE Validation**
Runs the pre-trained hybrid system on the LIVE dataset to validate the PLCC/SROCC metrics reported in the paper.

```bash
python scripts/1_reproduce_live.py

```

**Experiment 2: CSIQ Generalization**
Runs the auto-calibration and training routine on the CSIQ dataset (Table 2).

```bash
python scripts/2_reproduce_csiq.py

```

**Experiment 3: Complexity Audit**
Analyzes the internal structure of the trained SVRs to prove the reduction in model complexity.

```bash
python scripts/3_check_complexity.py

```

**Experiment 4: Full Comparative Benchmark**
Re-trains and optimizes BRISQUE from scratch and compares it against ARQUE side-by-side.

```bash
python scripts/4_benchmark_vs_brisque.py

```

---

## 📜 Citation

If you use this code or the ARQUE framework in your research, please cite:

```bibtex
@article{Frias2025ARQUE,
  title={ARQUE: A Hybrid Multi-Expert Framework for No-Reference Image Quality Assessment Using Curvature Analysis},
  author={Frias, Diego and Coelho, Leandro and Fonseca, Vagner},
  journal={Submitted to ELCVIA},
  year={2025}
}

```

---

📄 **License**
This project is licensed under the MIT License - see the LICENSE file for details.

```

### O que mudou e por quê?

1.  **Números Atualizados (Tabela 1):** Substituí os valores antigos (0.954/0.874) pelos reais e reproduzíveis de hoje (0.947/0.907). Isso dá credibilidade total.
2.  **Terminology:** Troquei "Divide and Conquer" por **"Distortion-Specific Decomposition"** e mencionei o **Soft-Voting**. Isso alinha o README com o tom técnico atualizado do Paper.
3.  **Script 4:** Adicionei o `4_benchmark_vs_brisque.py` na lista. Esse é o script mais importante para convencer um revisor cético, pois mostra o comparativo lado a lado.
4.  **BRISQUE (Optimized):** Deixei claro na tabela que o BRISQUE foi "Optimized", para justificar por que o número dele é alto (0.907) e mostrar que fomos justos.

Agora seu repositório está **profissional, honesto e alinhado com o PDF**.

```
