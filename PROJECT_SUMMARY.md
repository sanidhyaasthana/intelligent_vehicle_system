# Risk-Aware Vehicle Authorization: A Multi-Modal Fusion Framework

## Problem Statement

Vehicle authentication systems must balance security and usability across diverse operational conditions. Existing single-modal approaches either sacrifice security for convenience or become brittle under environmental degradation (low lighting, occlusion, GPS noise). This work addresses the need for risk-aware biometric fusion that adapts to input quality and location uncertainty.

## Core Contribution

We propose a multi-modal authorization framework combining:

1. **Adaptive-Margin Face Verification**: ArcFace with quality-aware margins that tighten for high-quality images and relax for degraded images, supporting stable verification across image quality variations.

2. **Probabilistic Geofence Modeling**: A learned trust model that outputs location confidence rather than binary decisions, capturing GPS uncertainty and enabling fine-grained risk assessment.

3. **Fusion-Based Decision Engine**: A risk fusion system combining face and geo trust scores with configurable thresholds for authorization (ALLOW/WARN/BLOCK) decisions.

## Key Contributions

- **Margin Adaptation**: Empirical validation that quality-aware margins reduce false rejection rates under natural degradation while maintaining security margins for high-quality images.

- **Probabilistic Geofencing**: Learned geofence model compared to hard boundaries shows improved modeling of spatial uncertainty and temporal patterns in location data.

- **Controlled Multi-Component Evaluation**: Four fusion configurations (fixed/adaptive face × hard/probabilistic geo) enabling fair ablation and demonstrating synergistic effects of adaptation.

- **Reproducible Experimental Framework**: Configuration-driven experiments with fixed random seeds, metrics (FAR/FRR/EER/AUC), and automated result aggregation.

- **Modular Implementation**: Extensible architecture supporting component replacement and alternative algorithms without modifying evaluation infrastructure.

## System Architecture

Face Verification (ResNet-50 + ArcFace) outputs T_face ∈ [0,1] with quality-aware margins. Geofence Verification (MLP Trust Model) outputs T_geo ∈ [0,1]. Fusion Engine computes risk R = α(1-T_face) + β(1-T_geo) and makes decisions (ALLOW/WARN/BLOCK) based on configurable thresholds.

## Experimental Outcomes

System evaluation generates per-component metrics (FAR, FRR, EER, ROC/AUC) and fusion comparison metrics (attack success rate, legitimate rejection rate, risk distributions) across four configurations. All metrics aggregated in system_comparison.csv with identical train/val/test splits across experiments.

## Reproducibility

Experiments are config-driven and deterministic. Random seeds are fixed across NumPy, PyTorch, and Python for all runs. Model paths and hyperparameters are specified in YAML configuration files; no hardcoding in source code. Identical dataset splits (train 70%, val 15%, test 15%) are maintained across all component and fusion experiments.
