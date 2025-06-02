# Federated Learning for Alzheimer’s MRI Classification

**Author:** Mohammad Karami
**Date:** December 16, 2024

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset and Task](#dataset-and-task)
3. [Centralized Baseline](#centralized-baseline)
4. [Phases of the Study](#phases-of-the-study)

   * [Phase 1: Basic FL Algorithms (IID)](#phase-1-basic-fl-algorithms-iid)
   * [Phase 2: Non-IID & Byzantine Attacks](#phase-2-non-iid--byzantine-attacks)
   * [Phase 3: Advanced FL Methods (IID)](#phase-3-advanced-fl-methods-iid)
   * [Phase 4: Advanced Methods under Non-IID](#phase-4-advanced-methods-under-non-iid)
   * [Phase 5: Enhancements for Non-IID](#phase-5-enhancements-for-non-iid)
   * [Phase 6: FLGuard against Byzantine Attacks](#phase-6-flguard-against-byzantine-attacks)
   * [Phase 7: Future Integration & Outlook](#phase-7-future-integration--outlook)
5. [Setup & Requirements](#setup--requirements)
6. [How to Run the Notebook](#how-to-run-the-notebook)
7. [Key Results Summary](#key-results-summary)
8. [Directory Structure (Suggested)](#directory-structure-suggested)
9. [Citation](#citation)

---

## Project Overview

This project implements and compares multiple **Federated Learning (FL)** algorithms—ranging from classic baselines (FedAvg, FedProx, FedADMM) to more advanced methods (FedNova, FedAdam, FedDWA, SCAFFOLD, FedBN)—for the task of **Alzheimer’s MRI classification**. We also study realistic non-IID splits (Label Skew, Dirichlet partitions) and evaluate robustness against **Byzantine (malicious) clients**. Finally, we introduce **FLGuard**, a robust aggregation scheme, and outline a future integrated approach combining FedProx + FedBN + FLGuard.

**Core Contributions**:

* Comparison of FL baselines under IID and Non-IID scenarios (with and without adversarial clients).
* Extended study of advanced FL algorithms (FedNova, FedAdam, FedDWA, SCAFFOLD, FedBN).
* Hyperparameter‐tuned enhancements (especially for FedBN) to nearly recover IID‐level performance under Non-IID splits.
* Integration of a robust aggregation strategy (FLGuard) to counter Byzantine attacks.
* Proposal for a unified pipeline combining FedProx, FedBN, and FLGuard for simultaneous Non-IID and adversarial resilience.

---

## Dataset and Task

* **Dataset**: T1‐weighted MRI scans labeled into four Alzheimer’s impairment categories:

  1. Mild Impairment
  2. Moderate Impairment
  3. No Impairment
  4. Very Mild Impairment

* **Preprocessing**:

  * Each MRI is resized to **224 × 224** pixels.
  * We use a **ResNet50** backbone, unfreezing the last 20 layers for fine‐tuning.
  * Training/validation/test splits are defined according to each FL scenario (IID or Non-IID partitioning among clients).

* **Clinical Significance**: Early detection of Alzheimer’s impairment level is crucial for timely intervention. FL enables multi‐institutional collaboration without sharing raw patient scans—preserving privacy.

---

## Centralized Baseline

Before federated experiments, a **centralized model** is trained on aggregated data:

* **Model**: ResNet50 (last 20 layers unfrozen)
* **Optimizer**: Adam (learning rate = 1e-4)
* **Epochs**: 10
* **Result**:

  * **Test Accuracy**: ≈ 95.47 %
  * Confusion matrix shows high precision/recall across all four classes.

This centralized performance serves as an upper bound. All FL experiments aim to approach this benchmark without requiring data pooling.

---

## Phases of the Study

### Phase 1: Basic FL Algorithms (IID)

1. **FedAvg**

   * Local clients train with standard SGD/Adam on their IID‐split data.
   * Server averages model weights each round.
   * **Result (IID)**: Test accuracy ≈ 94.68 %.

2. **FedProx**

   * Adds a proximal term $\frac{\mu}{2}\|w_i - w^{(t)}\|^2$ to each client’s loss to stabilize updates.
   * **Result (IID)**: Test accuracy ≈ 95.47 %, slightly surpassing FedAvg.

3. **FedADMM**

   * Uses an ADMM‐style consensus formulation: adds dual variables and a penalty term $\frac{\rho}{2}\sum_i\|w_i - w\|^2$.
   * More sensitive to $\rho$ (penalty hyperparameter).
   * **Result (IID)**: Test accuracy ≈ 79.75 % (underperformed due to non-optimized $\rho$).

**Implementation Details**

* **Model**: ResNet50 (unfreeze last 20 layers).
* **Local optimizer**: Adam (lr = 1e-4) + Batch Size = 8 (per client).
* **Clients**: Data split equally among $N$ clients (IID).
* **Communication rounds**: 10 – 20 (depending on experiment).
* **Local epochs per round**: 5.

> **Key Insight (IID):**
>
> * FedProx slightly outperforms FedAvg.
> * FedADMM underperforms without careful tuning.

---

### Phase 2: Non-IID Scenarios & Byzantine Attacks

We introduce two Non-IID partitions and a single Byzantine (malicious) client:

1. **Label Skew**

   * Each client’s local data is biased toward a subset of classes (e.g., one client has mostly “Mild” and “Moderate” scans).
   * **FedAvg**: Test acc. ≈ 93.04 %
   * **FedProx**: Test acc. ≈ 92.81 %
   * **FedADMM**: Test acc. ≈ 74.04 %

2. **Dirichlet Split (α = 0.5)**

   * Random heterogeneous allocation of labels across clients via a Dirichlet distribution.
   * **FedAvg**: Test acc. ≈ 84.21 %
   * **FedProx**: Test acc. ≈ 89.37 %
   * **FedADMM**: Test acc. ≈ 69.98 %

3. **Byzantine Attack**

   * One client flips/negates its model updates (simulated adversary).
   * **Result (any algorithm without defense)**: Test acc. ≈ 35 % (model collapses).

> **Key Insights (Phase 2):**
>
> * Under Label Skew, FedAvg/FedProx lose only a few points (< 2 %).
> * Under Dirichlet, FedProx significantly outperforms FedAvg (89.37 % vs 84.21 %).
> * FedADMM suffers most under both Non-IID splits when $\rho$ is not tuned.
> * All three collapse under a single Byzantine adversary—motivating robust aggregation (Phase 6).

---

### Phase 3: Advanced FL Methods (IID)

We evaluate five recently proposed FL algorithms under IID data:

1. **FedNova** (“Normalized Averaging”)

   * Normalizes client updates by the effective number of local steps $\tau_i$.
   * **IID Result**: Test acc. ≈ 96.01 % (top performer among baselines).

2. **FedAdam** (Adaptive server update)

   * Server applies Adam‐style momentum on aggregated gradients $(m_t,v_t)$.
   * **IID Result**: Test acc. ≈ 46.05 % (poor performance → requires careful tuning).

3. **FedDWA** (Dynamic Weight Adjustment)

   * Server weights each client’s update by $\alpha_i = \frac{1/(L_i + \varepsilon)}{\sum_j 1/(L_j + \varepsilon)}$, where $L_i$ = client $i$’s local loss.
   * **IID Result**: Test acc. ≈ 95.23 %.

4. **SCAFFOLD** (Control Variates)

   * Introduces control variates $c_i$ to correct client drift:

     $$
       \Delta c_i = c - c_i, \quad c \leftarrow c + \tfrac{1}{N}\sum_i \Delta c_i, \quad c_i \leftarrow c.
     $$
   * **IID Result**: Test acc. ≈ 88.66 %.

5. **FedBN** (Local BatchNorm)

   * Excludes BatchNorm mean/variance from global averaging—each client keeps its own BN stats.
   * **IID Result**: Test acc. ≈ 96.25 % (best among all IID experiments).

> **Key Insights (Phase 3):**
>
> * **FedBN** (96.25 %) and **FedNova** (96.01 %) both surpass FedProx (95.47 %).
> * **FedDWA** (95.23 %) is competitive.
> * **FedAdam** (46.05 %) and **SCAFFOLD** (88.66 %) underperform without extra tuning.

---

### Phase 4: Advanced Methods under Non-IID

We test the Phase 3 algorithms in the same two Non-IID settings (Label Skew, Dirichlet):

1. **FedNova**

   * Label Skew → Test acc. ≈ 90.62 %
   * Dirichlet → Test acc. ≈ 85.77 %

2. **FedAdam**

   * Label Skew → Test acc. ≈ 45.27 %
   * Dirichlet → Test acc. ≈ 48.79 %

3. **SCAFFOLD**

   * Label Skew → Test acc. ≈ 86.00 %
   * Dirichlet → Test acc. ≈ 84.05 %

4. **FedBN**

   * Label Skew → Test acc. ≈ 85.07 %
   * Dirichlet → Test acc. ≈ 87.33 %

5. **FedDWA**

   * Label Skew → Test acc. ≈ 83.58 %
   * Dirichlet → Test acc. ≈ 82.41 %

> **Comparative Insights (Phase 4):**
>
> * None of the advanced methods consistently outperforms a well-tuned FedProx (89.37 % under Dirichlet).
> * **FedNova** (90.62 % Label Skew) and **FedBN** (87.33 % Dirichlet) retain some advantage over FedAvg.
> * **FedAdam** remains ineffective under Non-IID without specialized tuning.
> * **SCAFFOLD** and **FedDWA** are moderate but do not surpass FedProx.

---

### Phase 5: Enhancements for Non-IID

We focus on **tuning FedBN** (and propose similar enhancements for SCAFFOLD):

* **Increased local epochs & global rounds**: e.g., 10 epochs per client + 20 rounds.
* **Lowered learning rate**: 1e-5 (instead of 1e-4) for smoother convergence.
* **Larger batch size**: 16 → more stable gradient estimates.
* **Unfreeze more ResNet layers**: e.g., last 40 layers (instead of 20) for deeper fine-tuning.

**Enhanced FedBN Results**:

* Label Skew → Test acc. ≈ 95 %
* Dirichlet → Test acc. ≈ 96 %

These numbers nearly match IID performance, demonstrating that Non-IID is not a “hard cap”—it can be mitigated through careful hyperparameter tuning and deeper client adaptation.

> **Potential SCAFFOLD Enhancement**:
>
> * Similar increase in epochs/rounds + lower lr + better control‐variates update schedule → improved stability. Though we did not run this variant, the logic from FedBN suggests SCAFFOLD can also improve under Non-IID.

---

### Phase 6: FLGuard against Byzantine Attacks

We implement **FLGuard** (Cao et al., CCS ’21), a robust aggregation strategy:

* **Core Idea**: For each communication round, examine pairwise distances among client model updates. Identify outliers (those that are “too far” from the majority) and exclude or downweight them before averaging.
* **Scenario**: IID data + 1 malicious client that negates its update.
* **Without FLGuard**: Test acc. ≈ 35 %.
* **With FLGuard**: Test acc. ≈ 90.85 %

> **Key Insight (Phase 6):**
>
> * Even strong FL algorithms collapse under a single Byzantine when using naive averaging.
> * FLGuard restores high accuracy by filtering the adversarial update—crucial for any real-world FL deployment where clients might be compromised.

---

### Phase 7: Future Integration & Outlook

We propose an **integrated pipeline** combining the best elements:

1. **FedProx** (local proximal regularization)
2. **FedBN** (keep BatchNorm statistics local)
3. **FLGuard** (robust aggregation)

**Conceptual Steps**:

1. **Client‐side**: Each client optimizes

   $$
     \min_{w_i} \; f_i(w_i) \;+\; \frac{\mu}{2}\|\,w_i - w^{(t)}\|^2
   $$

   where $w^{(t)}$ = global model weights at round $t$. BatchNorm layers keep local mean/var.

2. **Server‐side**:

   * Collect $\{w_i^{(t+1)}\}_{i=1}^N$.
   * Run FLGuard to detect outlier updates (potentially Byzantine).
   * Average (or weighted‐average) only the “honest” updates to produce $w^{(t+1)}$.

**Expected Benefits**:

* **Non-IID Robustness**: FedProx + FedBN can achieve near-IID levels under Label Skew/Dirichlet (as shown by Enhanced FedBN).
* **Adversarial Resilience**: FLGuard protects against malicious clients, maintaining accuracy even if some participants are compromised.
* **Unified Pipeline**: Offers a practical, end-to-end solution for real-world FL tasks, where data is non-IID and some clients may act adversarially.

**Next Steps**:

1. Implement and tune the combined FedProx + FedBN + FLGuard pipeline.
2. Evaluate under Label Skew + Byzantine, Dirichlet + Byzantine to verify simultaneous resilience.
3. Compare against standalone baselines and report a final set of results toward a comprehensive publication.

---

## Setup & Requirements

This project is organized as a single Jupyter notebook (`Alzheimer_FL_Study.ipynb`). To reproduce all experiments:

1. **Operating System**: Ubuntu 20.04 LTS (recommended) or any Linux/macOS.
2. **Python**: 3.8 – 3.10 (create a virtual environment to isolate dependencies).
3. **GPU**: NVIDIA GPU (≥ 8 GB VRAM) for reasonable training times, with CUDA 11.x.
4. **Dependencies**:

   * `torch` (≥ 1.10)
   * `torchvision` (≥ 0.11)
   * `numpy`
   * `pandas`
   * `scikit-learn`
   * `matplotlib`
   * `seaborn`  (optional, only for plotting)
   * `tqdm`  (progress bars)
   * `h5py`  or similar (if MRI data stored in HDF5)
   * `scipy`
   * `albumentations`  or `torchvision.transforms` (for data augmentations)
   * `jupyter` /notebook or Jupyter Lab

Create a new environment and install:

```bash
python3 -m venv venv
source venv/bin/activate      # (Windows: venv\Scripts\activate)
pip install --upgrade pip

# Core ML libraries
pip install torch torchvision
pip install numpy pandas scikit-learn matplotlib tqdm scipy

# (Optional) For visualizations
pip install seaborn

# If your MRI data is in HDF5 format
pip install h5py

# If using Albumentations for augmentation
pip install albumentations

# Jupyter
pip install jupyterlab
```

---

## How to Run the Notebook

1. **Prepare the Dataset**

   * Place preprocessed MRI images in a folder, e.g., `data/alzheimer_mri/`.
   * The notebook assumes a structure:

     ```
     data/
       alzheimer_mri/
         class_mild/
         class_moderate/
         class_no_impairment/
         class_very_mild/
     ```
   * Each subfolder contains `.png` or `.jpg` MRI scans resized to **224 × 224** prior to running.

2. **Launch Jupyter**

   ```bash
   cd <project-root>
   jupyter lab
   ```

   or

   ```bash
   jupyter notebook
   ```

3. **Open** `Alzheimer_FL_Study.ipynb`

   * The notebook is structured in sections matching the report’s phases.
   * Toggle cell execution in order:

     1. **Data loading & preprocessing**
     2. **Centralized baseline**
     3. **Phase 1 (FedAvg, FedProx, FedADMM)**
     4. **Phase 2 (Non-IID & Byzantine)**
     5. **Phase 3 (FedNova, FedAdam, FedDWA, SCAFFOLD, FedBN)**
     6. **Phase 4 (Advanced Non-IID)**
     7. **Phase 5 (Enhanced FedBN)**
     8. **Phase 6 (FLGuard)**
     9. **Phase 7 (Sketch of integrated approach)**

> **Note**: Because this is a single notebook (not modular), each section will define any required models, optimizers, or hyperparameters locally. Simply follow the notebook’s instructions and run cells sequentially.

---

## Key Results Summary

| Phase / Method                              | IID Accuracy (%) | Label Skew (%) | Dirichlet (%) | Byzantine (%) |
| :------------------------------------------ | :--------------: | :------------: | :-----------: | :-----------: |
| **Centralized (ResNet50, Adam, 10 epochs)** |       95.47      |        —       |       —       |       —       |
| **FedAvg**                                  |       94.68      |      93.04     |     84.21     |       35      |
| **FedProx**                                 |       95.47      |      92.81     |     89.37     |       35      |
| **FedADMM**                                 |       79.75      |      74.04     |     69.98     |       35      |
| **FedNova**                                 |       96.01      |      90.62     |     85.77     |      N/A      |
| **FedAdam**                                 |       46.05      |      45.27     |     48.79     |      N/A      |
| **FedDWA**                                  |       95.23      |      83.58     |     82.41     |      N/A      |
| **SCAFFOLD**                                |       88.66      |      86.00     |     84.05     |      N/A      |
| **FedBN**                                   |       96.25      |      85.07     |     87.33     |      N/A      |
| **Enhanced FedBN**                          |         —        |      \~ 95     |     \~ 96     |      N/A      |
| **FLGuard (IID + Byzantine)**               |         —        |        —       |       —       |     90.85     |

* *FedProx* and *FedBN* both match or exceed FedAvg under IID.
* Under Dirichlet, *FedProx* (89.37 %) and *Enhanced FedBN* (\~ 96 %) show the greatest resilience.
* *FedNova* is strong under IID but loses edge under Dirichlet (85.77 %).
* *FedAdam* consistently underperforms without fine-tuning.
* *SCAFFOLD* and *FedDWA* are moderate under Non-IID.
* *Enhanced FedBN* recovers near-IID performance (> 95 %) under both Label Skew and Dirichlet with more training/rounds.
* *FLGuard* recovers accuracy from \~ 35 % (naive averaging with 1 Byzantine) to \~ 90.85 %.

---

## Directory Structure (Suggested)

```
/
├── data/
│   └── alzheimer_mri/
│       ├── mild/                     ← MRI images for class = Mild
│       ├── moderate/                 ← MRI images for class = Moderate
│       ├── no_impairment/            ← MRI images for class = No Impairment
│       └── very_mild/                ← MRI images for class = Very Mild Impairment
│
├── Alzheimer_FL_Study.ipynb          ← Single‐notebook implementation
├── requirements.txt                  ← pip freeze > requirements.txt (optional)
├── README.md                         ← This file
└── LICENSE                           ← (e.g., MIT License)
```

> **Tip:** If later you modularize, you might split into:
>
> ```
> /models/         ← PyTorch model definitions (ResNet50 wrapper, custom layers)
> /utils/          ← Utility functions (data loader, metrics, plotting)
> /experiments/    ← Scripts to launch each Phase (phase1_iid.py, phase2_non_iid.py, etc.)
> Alzheimer_FL_Study.ipynb
> README.md
> ```
>
> But for now, the entire workflow lives in one notebook.

---

## Citation

If you use any part of this code or reproduce these experiments, please cite:

```
Karami, M. (2024). Federated Learning for Alzheimer MRI Classification: 
A Comparative Study of FedAvg, FedProx, and FedADMM (and advanced methods) 
under IID, Non-IID, and Byzantine Scenarios. December 16, 2024.
```

For related methodologies, you may also reference:

* McMahan et al., “Communication‐efficient learning of deep networks from decentralized data,” AISTATS 2017.
* Li et al., “FedProx: Federated Optimization in Heterogeneous Networks,” MLSys 2020.
* Zhang et al., “Stochastic primal‐dual ADMM for distributed nonconvex optimization,” IEEE TSP 2018.
* Wang et al., “FedNova: An Optimization Perspective and Trade‐offs of Federated Learning,” arXiv 2020.
* Reddi et al., “Adaptive Federated Optimization,” ICLR 2021.
* Wang et al., “Dynamic Weight Adjustment for Federated Learning,” arXiv 2020.
* Karimireddy et al., “SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,” ICML 2020.
* Li et al., “FedBN: Federated Learning on Non‐IID Features via Local Batch Normalization,” ICLR 2021.
* Cao et al., “FLGuard: Secure and Private Federated Learning,” CCS 2021.

---

*End of README*
