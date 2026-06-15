import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import DeepSpeedConfig
from dataclasses import dataclass

@dataclass
class CurriculumStage:
    name: str
    dataset_path: str
    mix_ratio: float
    epochs: int
    learning_rate: float

class ContinualPreTrainer:
    """Continuous pre-training with catastrophic forgetting mitigation."""
    def __init__(self, base_model: str, domain: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.domain = domain
        self.fisher_matrices = {}

    def compute_fisher(self, dataloader, num_samples=1000):
        """Compute Fisher Information Matrix for EWC."""
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters()}
        self.model.eval()
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            output = self.model(**batch)
            loss = output.loss
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        for n in fisher:
            fisher[n] /= num_samples
        self.fisher_matrices[self.domain] = fisher

    def ewc_loss(self, ewc_lambda: float = 5000.0):
        """Elastic Weight Consolidation penalty."""
        loss = 0.0
        for domain, fisher in self.fisher_matrices.items():
            for n, p in self.model.named_parameters():
                ref = self.reference_params[n]
                loss += (fisher[n] * (p - ref) ** 2).sum()
        return ewc_lambda * loss

    def train_stage(self, stage: CurriculumStage, ds_config: dict):
        dataset = self.load_and_filter(stage.dataset_path)
        mixed = self.mix_with_replay(dataset, stage.mix_ratio)
        trainer = DeepSpeedTrainer(
            model=self.model, config=ds_config,
            train_dataset=mixed, lr=stage.learning_rate,
            extra_loss_fn=self.ewc_loss
        )
        trainer.train(num_epochs=stage.epochs)