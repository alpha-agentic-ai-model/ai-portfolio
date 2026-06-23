"""Vision Transformer for Medical Image Classification."""
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 5
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    class_names: list[str] = field(default_factory=lambda: [
        "Normal", "Pneumonia", "Cardiomegaly", "Effusion", "Nodule"
    ])


class PatchEmbedding:
    """Convert image into patch embeddings."""

    def __init__(self, config: ModelConfig):
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.embed_dim = config.embed_dim
        # Simulated projection weights
        self.proj_weight = np.random.randn(
            config.patch_size * config.patch_size * 3, config.embed_dim
        ) * 0.02

    def forward(self, images: np.ndarray) -> np.ndarray:
        batch_size = images.shape[0]
        h = w = images.shape[2]
        p = self.patch_size
        n_h, n_w = h // p, w // p

        # Extract patches
        patches = images.reshape(batch_size, 3, n_h, p, n_w, p)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(batch_size, n_h * n_w, -1)

        # Project to embedding dimension
        embeddings = patches @ self.proj_weight
        return embeddings


class MultiHeadAttention:
    """Multi-head self-attention with attention map storage."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_map: Optional[np.ndarray] = None

        # Simulated weights
        self.qkv_weight = np.random.randn(embed_dim, 3 * embed_dim) * 0.02
        self.out_weight = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, N, C = x.shape
        qkv = (x @ self.qkv_weight).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = self._softmax(attn)
        self.attention_map = attn.copy()

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return out @ self.out_weight

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class TransformerBlock:
    """Single transformer encoder block."""

    def __init__(self, config: ModelConfig):
        self.attn = MultiHeadAttention(
            config.embed_dim, config.num_heads, config.dropout
        )
        self.mlp_weight1 = np.random.randn(
            config.embed_dim, int(config.embed_dim * config.mlp_ratio)
        ) * 0.02
        self.mlp_weight2 = np.random.randn(
            int(config.embed_dim * config.mlp_ratio), config.embed_dim
        ) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Self-attention with residual
        x = x + self.attn.forward(self._layer_norm(x))
        # MLP with residual and GELU
        h = self._layer_norm(x) @ self.mlp_weight1
        h = self._gelu(h) @ self.mlp_weight2
        return x + h

    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


class GradCAM:
    """Generate attention-based explanation maps."""

    def __init__(self, model: "MedicalViT"):
        self.model = model

    def generate(self, image: np.ndarray, target_class: int) -> np.ndarray:
        # Use last layer attention weights as proxy for GradCAM
        attn = self.model.get_attention_map()
        if attn is None:
            return np.zeros((14, 14))

        # Average across heads, take CLS token attention
        cls_attn = attn[0].mean(axis=0)[0, 1:]  # skip CLS-to-CLS
        grid_size = int(math.sqrt(len(cls_attn)))
        cam = cls_attn.reshape(grid_size, grid_size)

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class MedicalViT:
    """Vision Transformer for Medical Image Classification."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.patch_embed = PatchEmbedding(self.config)
        self.cls_token = np.random.randn(1, 1, self.config.embed_dim) * 0.02

        num_patches = self.patch_embed.num_patches
        self.pos_embed = np.random.randn(
            1, num_patches + 1, self.config.embed_dim
        ) * 0.02

        self.blocks = [
            TransformerBlock(self.config)
            for _ in range(self.config.num_layers)
        ]
        self.head_weight = np.random.randn(
            self.config.embed_dim, self.config.num_classes
        ) * 0.02

    def forward(self, images: np.ndarray) -> np.ndarray:
        B = images.shape[0]
        x = self.patch_embed.forward(images)
        cls_tokens = np.tile(self.cls_token, (B, 1, 1))
        x = np.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block.forward(x)

        cls_output = x[:, 0]
        logits = cls_output @ self.head_weight
        return logits

    def predict(self, images: np.ndarray) -> list[dict]:
        logits = self.forward(images)
        probs = self._softmax(logits)
        results = []
        for i in range(len(images)):
            pred_class = int(np.argmax(probs[i]))
            results.append({
                "class": self.config.class_names[pred_class],
                "confidence": float(probs[i, pred_class]),
                "all_probs": {
                    name: float(probs[i, j])
                    for j, name in enumerate(self.config.class_names)
                },
            })
        return results

    def get_attention_map(self) -> Optional[np.ndarray]:
        if self.blocks:
            return self.blocks[-1].attn.attention_map
        return None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    config = ModelConfig(num_layers=2, num_heads=4)
    model = MedicalViT(config)
    images = np.random.randn(2, 3, 224, 224).astype(np.float32)

    predictions = model.predict(images)
    for i, pred in enumerate(predictions):
        print(f"Image {i}: {pred['class']} ({pred['confidence']:.3f})")

    cam = GradCAM(model)
    heatmap = cam.generate(images[0:1], target_class=0)
    print(f"GradCAM heatmap shape: {heatmap.shape}")
