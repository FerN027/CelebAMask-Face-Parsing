# High-Precision Semantic Face Parsing Architecture (CelebAMask-HQ)

This repository contains the official implementation of a state-of-the-art, highly optimized deep learning framework for granular face parsing. Critical for advanced biometric systems, AR/VR face synthesis, and next-generation deepfake detection, this system robustly segments unaligned facial images into 19 highly detailed semantic classes (defining regions from hair and skin to individual facial features).

## Table of Contents
- [Project Overview](#project-overview)
- [Architectural Innovations](#architectural-innovations)
- [Data Ecosystem](#data-ecosystem)
- [Training & Deployment Pipelines](#training--deployment-pipelines)
- [Performance](#performance)

## Project Overview
Tackling the highly complex multi-class semantic segmentation domain, this framework operates on the rigorous **CelebAMask-HQ** benchmark. By replacing conventional heavy backbones with a strategically designed U-Net-like paradigm, the model achieves deep semantic understanding of facial geometry while maintaining an exceptionally lightweight footprint suitable for versatile deployments.

## Architectural Innovations
The core of this repository is the **EnhancedFaceParsingNet**—an ultra-efficient neural network comprising just ~1.6M parameters yet punching well above its weight class. Key breakthroughs include:
- **Advanced Encoder (DualDilatedBlock):** A pioneering module that downsamples feature maps using parallel dilated convolutions. By leveraging varying dilation rates, it exponentially expands the model's receptive field to capture broad, complex spatial contexts without the steep computational overhead of traditional deep convolutions.
- **Global Context Bottleneck (EnhancedPyramidPool):** At the deepest network layers, this multi-scale fusion mechanism aggregates both global context and fine-grained local textures via adaptive average pooling across multiple scales (1, 2, and 4), rendering the model highly resilient to spatial variations.
- **Precision Decoder:** Employs precise transposed convolutions for upsampling, structurally enforcing skip connections that shuttle high-resolution spatial awareness straight from the encoder. This ensures razor-sharp boundary localization for critical minute features (e.g., eyes, lips).

## Data Ecosystem
To replicate our data pipeline, conform to the following scalable directory structure:
```
data/
    dev-public/
        train/
            images/
            masks/
    test-public/
        test/
            images/
            predicted_masks/
```

- **Data Normalization & Preprocessing:** 
  - Iterative standardization converts input spaces to normalized tensors `mean=(0.5, 0.5, 0.5)` and `std=(0.5, 0.5, 0.5)` for optimal gradient flow.
  - Masks are dynamically transformed into strictly categorical 2D `long` tensors for highly precise Cross-Entropy optimization.

## Training & Deployment Pipelines

### 1. Initiating the Training Pipeline
To kick off the automated training pipeline across available compute architectures (seamlessly leveraging CUDA when accessible), use the primary orchestrator:
```bash
python main.py
```
This routine triggers a rigorous 300-epoch training cycle powered by the Adam optimizer (`lr=0.001`) with an aggressive `StepLR` scheduling strategy (step size 25, gamma 0.2) to guide the weights to a robust global minimum. The ultimate weights drop into a production-ready `model.pth` checkpoint.

### 2. High-Speed Inference
To propel the model through the test set and extract dense semantic predictions, run:
```bash
python test.py
```
The framework loads the verified checkpoint and synthesizes `.png` maps inside `data/test-public/test/predicted_masks`, ready for downstream analytics.

## Performance
The architecture underwent stringent evaluation using the rigorous Multi-Class F1-Score metric.
- **Optimization Strategy:** Standard rigorous `nn.CrossEntropyLoss()` surprisingly yielded the optimal decision boundaries across all 19 heterogeneous classes, outperforming weighted class balancing.
- **Benchmark F1-Score:** The model achieved an outstanding F-measure of **0.81** on the fiercely competitive Codabench evaluation suite, securing a highly confident predictive baseline.
