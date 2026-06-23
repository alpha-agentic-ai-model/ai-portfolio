# Vision Transformer for Medical Image Classification

## Overview
A fine-tuned Vision Transformer (ViT) for multi-class medical image classification with attention-map explainability. Implements GradCAM visualization, class-balanced sampling, and ONNX export for edge deployment.

## Architecture
```
[DICOM Input] → [Preprocessor] → [ViT Encoder]
  |
[Attention Maps] → [Classification Head] → [GradCAM] → [Clinical Report]
```

## Tech Stack
PyTorch, Hugging Face, ONNX, GradCAM, MONAI, timm

## Key Features
- Production-ready implementation with error handling
- Comprehensive type annotations and documentation
- Modular architecture for easy extension
- Built for scalability and performance
