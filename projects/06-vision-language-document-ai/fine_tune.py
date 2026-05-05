import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer
from datasets import load_dataset
import wandb


def setup_model(model_name='microsoft/Florence-2-large'):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        lora_dropout=0.05,
        task_type='VL_SEQ2SEQ',
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
    return model, processor


class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.processor(
            images=item['image'],
            text='Extract all fields as JSON',
            return_tensors='pt',
        )
        labels = self.processor.tokenizer(
            item['structured_output'],
            return_tensors='pt', padding='max_length', max_length=512,
        )
        inputs['labels'] = labels['input_ids']
        return {k: v.squeeze(0) for k, v in inputs.items()}


def compute_field_accuracy(predictions, references):
    correct = total = 0
    for pred, ref in zip(predictions, references):
        for key in ref:
            total += 1
            if key in pred and pred[key] == ref[key]:
                correct += 1
    return {'field_accuracy': correct / total if total > 0 else 0}


def train():
    wandb.init(project='document-vlm', name='florence2-lora-ft')
    model, processor = setup_model()
    dataset = load_dataset('custom_invoices', split='train')
    train_ds = DocumentDataset(dataset.select(range(0, 8000)), processor)
    eval_ds = DocumentDataset(dataset.select(range(8000, 10000)), processor)

    args = TrainingArguments(
        output_dir='./checkpoints', num_train_epochs=5,
        per_device_train_batch_size=4, gradient_accumulation_steps=8,
        learning_rate=2e-4, fp16=True,
        evaluation_strategy='steps', eval_steps=500,
        save_steps=500, logging_steps=100, report_to='wandb',
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_field_accuracy,
    )
    trainer.train()
    model.save_pretrained('./florence2-doc-understanding')


if __name__ == '__main__':
    train()
