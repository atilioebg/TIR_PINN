---
description: "Guidelines and Mathematical Standards for Q1 LaTeX Manuscript Drafting (IEEE TGRS / RSE)"
---

# Q1 LATEX MANUSCRIPT DRAFTING RULES (IEEE TGRS / RSE)

## 1. Mathematical Rigor & Notation Standards
* **Vectors & Matrices:** Always use boldface for spectral vectors and parameter matrices (`\bm{\epsilon}`, `\bm{L}_{\text{obs}}`, `\bm{W}`). Scalars must be standard italicized math font ($T_s, \lambda, \tau$).
* **Sobolev Space Representation:** Formally designate the discrete spectral Sobolev space as $W^{2,1}(\Omega)$ or $W^{2,2}(\Omega)$, explicitly defining forward finite differences for first derivatives ($\nabla \bm{\epsilon}$) and second derivatives ($\nabla^2 \bm{\epsilon}$).
* **Radiative Transfer Consistency:** Always adhere to the canonical Lambertian clear-sky RTE:
  $$L_{\text{obs}}(\lambda) = \tau(\lambda)\epsilon(\lambda)B(\lambda, T_s) + L_{\text{up}}(\lambda) + \tau(\lambda)[1 - \epsilon(\lambda)]L_{\text{down}}(\lambda)$$
* **Units:** Use standard SI remote sensing units ($[\text{W}\cdot\text{m}^{-2}\cdot\text{sr}^{-1}\cdot\mu\text{m}^{-1}]$ for spectral radiance, $[\text{K}]$ for kinetic surface temperature, and $[\mu\text{m}]$ for wavelength).

## 2. Bibliographic Integrity & Strict Grounding
* **Zero Hallucination:** Every cited key in `\cite{...}` must exist in `manuscript/references.bib`.
* **Authentic DOIs:** Always maintain verified DOIs and authoritative JPL/NASA publication IDs for ATBDs.
* **Literature Grounding:** Anchor all comparative assertions to the 12 verified project references.

## 3. Structural Standards for IEEE TGRS
* **Section Hierarchy:**
  - `\section{Introduction}`: Gap definition (ill-posed inverse problem, MMD breakdown, lack of shape preservation in DL).
  - `\section{Physical Principles and Radiative Transfer Baseline}`: RTE, Planck's law, and Kirchhoff's law.
  - `\section{Physics-Informed Architecture with Sobolev Shape Regularization}`: Multi-head topology, hard activations, discrete Sobolev loss derivation.
  - `\section{Experimental Methodology and Golden Target Protocol}`: HyTES (256 bands), ASTER (5 bands), clean-room Golden Target endmember isolation.
  - `\section{Experimental Results and Performance Analysis}`: Quantitative tables, Figure 1 (shape recovery), Figure 2 (temperature error).
  - `\section{Discussion and Reviewer \#2 Critical Defense}`: Ablation analysis, physical interpretability, satellite operational scalability (NASA SBG / Copernicus LSTM).
  - `\section{Conclusion}`: Takeaways, open-source codebase release, and future orbital flightline deployment.
