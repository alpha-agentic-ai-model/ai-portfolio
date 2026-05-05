# Vision-Language Model for Document Understanding

Fine-tuned vision-language model extracting structured data from documents (invoices, receipts, contracts) using LoRA adapters on Florence-2 with 97% field extraction accuracy.

## Architecture
```
[Document Image] -> [Vision Encoder] -> [Cross-Attention]
                                              |
              [Language Decoder + LoRA] -> [Structured JSON]
```

## Tech Stack
- PyTorch, HuggingFace PEFT, Florence-2, LoRA, Weights & Biases
