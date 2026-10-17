# TIR-PINN: Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation

[![arXiv](https://img.shields.io/badge/arXiv-submit%2F8242913-B31B1B.svg)](https://arxiv.org/)
[![Target: IEEE TGRS](https://img.shields.io/badge/Target-IEEE%20TGRS-00629B.svg)](https://mtt.org/publications/tgrs/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![ECOSTRESS Spectral Library](https://img.shields.io/badge/Data-ECOSTRESS%20%7C%20ASTER-blue.svg)](https://speclib.jpl.nasa.gov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official Implementation of the Research Manuscript:**  
> **"Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors"**  
> **Authors:** Atilio Grondona¹*, Bijeesh Kozhikkodan Veettil², Luis Morales-Salinas³  
> ¹ *State Center of Research on Remote Sensing and Meteorology (CEPSRM), Federal University of Rio Grande do Sul (UFRGS), Brazil*  
> ² *Faculty of Applied Sciences, Van Lang University, Ho Chi Minh City, Vietnam*  
> ³ *Laboratorio de Investigación en Ciencias Ambientales (LARES), Universidad de Chile, Santiago, Chile*  
> \* *Corresponding Author: atilio.grondona@ufrgs.br*

---

## 📌 Executive Summary

Temperature and Emissivity Separation (TES) in Thermal Infrared (TIR) remote sensing is a canonical **ill-posed inverse problem**: an orbital sensor observing $N$ spectral channels measures $N$ radiances, but must resolve $N + 1$ unknowns (one surface kinetic temperature $T_s$ plus $N$ band-averaged spectral emissivities $\varepsilon_i$).

Classical deterministic TES algorithms (e.g., Normalized Emissivity Method, Alpha-Derived Emissivity, Temperature-Independent Spectral Indices) resolve this underdetermination by relying on empirical min-max emissivity difference (MMD) power-law relationships. However, these heuristics suffer from severe instability under atmospheric compensation errors, low spectral contrast, and high-dimensional hyperspectral configurations.

**TIR-PINN** introduces a deep physics-informed architecture that fundamentally redefines TES:
1. **Differentiable Radiative Transfer Equation (RTE):** Enforces Planck's radiation law and MODTRAN-derived atmospheric profiles directly within the computational graph as soft physics constraints.
2. **Sobolev $H^1$ Shape Regularization:** Couples standard $L^2$ spectral reconstruction loss with an $L^2$ gradient penalty ($\|
abla_\lambda \varepsilon - \nabla_\lambda \hat{\varepsilon}\|_{L^2}^2$), eliminating unphysical high-frequency spectral oscillations while preserving narrow diagnostic absorption features (e.g., quartz reststrahlen doublets at $8.62\,\mu\text{m}$).
3. **Dual-Scale Sensor Generalization:** Evaluated across both underconstrained multispectral platforms (**ASTER**, 5 thermal bands) and high-dimensional hyperspectral sensors (**HyTES**, 256 contiguous thermal channels).

---

## 🔬 System Architecture & Training Workflow

```mermaid
flowchart TD
    subgraph Data["1. Spectral Data Curation (ECOSTRESS)"]
        Raw["Raw ECOSTRESS / ASTER JHU Spectral Library
(>6,000 Laboratory Target Spectra)"]
        Clean["Mineral & Soil Filtering
Kirchhoff Law Conversion: ε(λ) = 1 - R(λ)"]
        Library["Curated Pure Endmember Library
(Quartz, Calcite, Kaolinite, Feldspar, Water)"]
        Raw --> Clean --> Library
    end

    subgraph Physics["2. Differentiable Atmospheric Radiative Transfer (RTE)"]
        MODTRAN["MODTRAN 5.4 Radiative Transfer LUTs
Atmospheric Profiles: Mid-Latitude Summer/Winter"]
        Ts["Kinetic Temperature Sampling: Ts ~ U(280 K, 330 K)"]
        Planck["Planck Blackbody Function: B(λ, Ts)"]
        RTE["High-Resolution Top-of-Atmosphere (TOA) Radiance:
L_TOA(λ) = τ(λ)·ε(λ)·B(λ, Ts) + L_up(λ) + τ(λ)·[1 - ε(λ)]·L_down(λ)"]
        
        Library & Ts & MODTRAN --> Planck --> RTE
    end

    subgraph Sensor["3. Sensor Spectral Response Function (SRF) Degradation"]
        SRF_HyTES["HyTES SRF Convolution (256 Bands, 7.5 - 12.0 μm)"]
        SRF_ASTER["ASTER SRF Convolution (5 Bands, 8.125 - 11.65 μm)"]
        Noise["Gaussian Sensor Noise Injection: NEDT ~ N(0, σ²)"]
        
        RTE --> SRF_HyTES --> Noise
        RTE --> SRF_ASTER --> Noise
    end

    subgraph PINN["4. Physics-Informed Dual-Head Architecture"]
        Input["Observed TOA Radiance: L_obs"]
        Encoder["Shared Multi-Layer Perceptron / Residual Backbone"]
        Head_T["Temperature Head: Regression -> T̂s"]
        Head_E["Emissivity Head: Softplus/Sigmoid Constrained -> ε̂(λ)"]
        
        Noise --> Input --> Encoder
        Encoder --> Head_T
        Encoder --> Head_E
    end

    subgraph Loss["5. Sobolev H¹ Physics-Informed Optimization"]
        Loss_Data["Data Reconstruction Loss: ||L_obs - L_pred||²"]
        Loss_Sobolev["Sobolev H¹ Shape Loss:
||ε - ε̂||² + α·||∇_λ ε - ∇_λ ε̂||²"]
        Loss_Physics["Differentiable RTE Residual Loss"]
        
        Head_T & Head_E --> Loss_Data & Loss_Sobolev & Loss_Physics
        Backprop["AdamW Optimizer + Cosine Annealing Learning Rate"]
        Loss_Data & Loss_Sobolev & Loss_Physics --> Backprop --> Encoder
    end
```

---

## 📐 Mathematical Formulation

### 1. Thermal Radiative Transfer Equation (RTE)
For a non-isothermal plane-parallel atmosphere under local thermodynamic equilibrium, the spectral radiance observed at sensor altitude $L_{\text{obs}}(\lambda)$ is formulated as:

$$L_{\text{obs}}(\lambda) = \tau(\lambda) \varepsilon(\lambda) B(\lambda, T_s) + L^{\uparrow}(\lambda) + \tau(\lambda) [1 - \varepsilon(\lambda)] L^{\downarrow}(\lambda)$$

where:
* $B(\lambda, T_s) = \frac{c_1}{\lambda^5 \left[ \exp\left(\frac{c_2}{\lambda T_s}\right) - 1 \right]}$ is Planck's blackbody law ($c_1 = 1.19104 \times 10^8\,\text{W}\cdot\mu\text{m}^4/(\text{m}^2\cdot\text{sr})$, $c_2 = 1.43877 \times 10^4\,\mu\text{m}\cdot\text{K}$).
* $\tau(\lambda)$: Total atmospheric path transmittance from surface to sensor.
* $L^{\uparrow}(\lambda)$: Atmospheric upwelling thermal path radiance.
* $L^{\downarrow}(\lambda)$: Atmospheric downwelling sky irradiance reflected by the surface.
* $\varepsilon(\lambda)$: Directional land surface emissivity.
* $T_s$: Kinetic surface temperature (K).

### 2. Sobolev $H^1$ Shape Regularization
Standard neural network loss functions ($L^2$ or MSE) treat spectral channels as independent, uncorrelated dimensions, leading to unphysical high-frequency "sawtooth" artifacts in the retrieved emissivity spectrum. 

TIR-PINN regularizes the solution space inside the **Sobolev space $W^{1,2}(\Omega) = H^1(\Omega)$**, penalizing both amplitude errors and directional spectral gradient deviations:

$$\mathcal{L}_{H^1} = \|\varepsilon - \hat{\varepsilon}\|_{L^2}^2 + \alpha_s \|\nabla_\lambda \varepsilon - \nabla_\lambda \hat{\varepsilon}\|_{L^2}^2 = \frac{1}{N} \sum_{i=1}^{N} (\varepsilon_i - \hat{\varepsilon}_i)^2 + \frac{\alpha_s}{N-1} \sum_{i=1}^{N-1} \left[ (\varepsilon_{i+1} - \varepsilon_i) - (\hat{\varepsilon}_{i+1} - \hat{\varepsilon}_i) \right]^2$$

where $\alpha_s = 0.35$ enforces physical smoothness while strictly preserving true mineral reststrahlen band curvatures.

### 3. Total Multi-Task Loss Objective
$$\mathcal{L}_{\text{total}} = \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}}(L_{\text{obs}}, \hat{L}_{\text{obs}}) + \lambda_T |T_s - \hat{T}_s| + \lambda_{\text{Sobolev}} \mathcal{L}_{H^1}(\varepsilon, \hat{\varepsilon}) + \lambda_{\text{phys}} \mathcal{L}_{\text{RTE}}$$

---

## 📊 Scientific Validation & Quantitative Benchmarks

The framework was benchmarked against the standard NASA/JPL Temperature and Emissivity Separation (TES) algorithm across 1,200 independent out-of-distribution test spectra from the ECOSTRESS library under varying atmospheric profiles (MODTRAN Mid-Latitude Summer, column water vapor $\omega \in [0.5, 3.5]\,\text{g/cm}^2$):

### Temperature & Emissivity Separation Performance

| Sensor Platform | Spectral Bands | Retrieval Algorithm | Temperature MAE (K) | Temperature RMSE (K) | Emissivity RMSE | Spectral Angle Mapper (SAM, rad) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ASTER (Multispectral)** | 5 bands | Classical TES (Gillespie et al.) | 1.39 K | 1.84 K | 0.0241 | 0.064 rad |
| **ASTER (Multispectral)** | 5 bands | Standard MLP (No Physics) | 0.82 K | 1.15 K | 0.0165 | 0.041 rad |
| **ASTER (Multispectral)** | **5 bands** | **TIR-PINN (Ours)** | **0.38 K** | **0.52 K** | **0.0084** | **0.019 rad** |
| **HyTES (Hyperspectral)** | 256 bands | Classical Hyperspectral TES | 1.12 K | 1.48 K | 0.0189 | 0.048 rad |
| **HyTES (Hyperspectral)** | 256 bands | Standard PINN ($L^2$ Only) | 0.49 K | 0.68 K | 0.0092 | 0.024 rad |
| **HyTES (Hyperspectral)** | **256 bands** | **TIR-PINN with Sobolev $H^1$** | **0.27 K** | **0.36 K** | **0.0041** | **0.012 rad** |

### Key Physical Breakthroughs:
* **80.5% Temperature Error Reduction:** On hyperspectral data, kinetic temperature error dropped from $1.39\text{ K}$ (classical TES) to **$0.27\text{ K}$**, well within the radiometric noise limit of the instrument.
* **Ablation of Sobolev Shape Regularization:** Adding the Sobolev $H^1$ gradient penalty decreased spectral angle error (SAM) by **50%** ($0.024 \to 0.012\text{ rad}$), preventing spectral inversion collapse across carbonate and silicate absorption doublets.

---

## 🖼️ High-Resolution Validation Figures

| Figure 1: Emissivity Recovery Across Mineral Classes | Figure 2: Temperature Inversion Error Distributions |
| :---: | :---: |
| ![Emissivity Comparison](fig1_emissivity_comparison.png) | ![Temperature Accuracy](fig2_temperature_accuracy.png) |
| *Preservation of Quartz 8.62 μm doublet and Kaolinite spectral slopes under 256-band HyTES and 5-band ASTER sensors.* | *Residual temperature error distribution (K) comparing Classical TES vs. Sobolev-regularized TIR-PINN.* |

---

## 📁 Repository Structure

```
TIR_PINN/
├── src/
│   ├── data_generation/          # MODTRAN & ECOSTRESS Ingestion
│   │   ├── curate_spectral_library.py  # Kirchhoff conversion & quality filtering
│   │   ├── automate_modtran.py         # MODTRAN 5.4 atmospheric LUT automation
│   │   └── sensor_convolution.py       # ASTER & HyTES SRF integration
│   │
│   ├── models/                   # Neural Architectures
│   │   ├── tir_pinn.py           # Multi-Head PINN with Differentiable RTE
│   │   ├── sobolev_loss.py       # Custom Sobolev H¹ Loss Function
│   │   └── classical_tes.py      # Gillespie Classical TES Baseline Benchmark
│   │
│   ├── training/                 # Model Training & Hyperparameter Tuning
│   │   ├── train_generalist.py   # Large-scale base model training
│   │   └── train_specialist.py   # Low-learning-rate fine-tuning on high-fidelity spectra
│   │
│   └── evaluation/               # Publication Metrics & Plots
│       ├── evaluate_test_set.py  # Table I & Table II metric calculation (MAE, RMSE, SAM)
│       └── generate_figures.py   # Publication-grade figure generation (300+ DPI)
│
├── manuscript/                   # Active Research Manuscript
│   ├── main.tex                  # Complete IEEE TGRS LaTeX source
│   ├── references.bib            # Curated BibTeX references
│   └── figures/                  # Vectorized PDF and high-res PNG manuscript figures
│
├── data/                         # Spectral and atmospheric profiles
│   ├── ecostress_curated/        # Preprocessed spectral endmembers (.npy / .parquet)
│   └── modtran_lut/              # Atmospheric transmittance & path radiance LUTs
│
├── outputs/                      # Model checkpoints and evaluation logs
├── fig1_emissivity_comparison.png# High-res publication Figure 1
├── fig2_temperature_accuracy.png # High-res publication Figure 2
├── requirements.txt              # Environment dependencies
└── README.md                     # Main Scientific & Technical Documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/atilioebg/TIR_PINN.git
cd TIR_PINN

# Create virtual environment and install requirements
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
pip install -r requirements.txt
```

### 2. Preprocess Spectral Library & Convolve Sensors
```bash
# Extract and filter ECOSTRESS spectra
python src/data_generation/curate_spectral_library.py --input-dir data/ecostress_raw --output-dir data/ecostress_curated

# Convolve with ASTER and HyTES SRFs
python src/data_generation/sensor_convolution.py --sensor all
```

### 3. Train TIR-PINN with Sobolev Regularization
```bash
# Train on HyTES (Hyperspectral)
python src/training/train_generalist.py --sensor hytes --sobolev-weight 0.35 --epochs 150

# Train on ASTER (Multispectral)
python src/training/train_generalist.py --sensor aster --sobolev-weight 0.20 --epochs 150
```

### 4. Evaluate Test Metrics & Generate Figures
```bash
# Compute Table I & II validation statistics
python src/evaluation/evaluate_test_set.py --checkpoint outputs/hytes_best.pt

# Generate publication-grade figures
python src/evaluation/generate_figures.py
```

---

## 📖 Citation

If you use this codebase or methodology in your research, please cite our preprint:

```bibtex
@article{grondona2026tirpinn,
  title={Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors},
  author={Grondona, At{'i}lio and Veettil, Bijeesh Kozhikkodan and Morales-Salinas, Luis},
  journal={arXiv preprint arXiv:submit/8242913 [physics.geo-ph]},
  year={2026},
  note={Under review at IEEE Transactions on Geoscience and Remote Sensing (TGRS)}
}
```

---

## ⚖️ License
This project is open-source under the [MIT License](LICENSE).
