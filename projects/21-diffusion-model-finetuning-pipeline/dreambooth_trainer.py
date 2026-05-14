"""Diffusion Model Fine-Tuning with DreamBooth & LoRA

End-to-end pipeline for fine-tuning Stable Diffusion models using DreamBooth
and LoRA adapters on custom image datasets.
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms
import wandb
import logging

logger = logging.getLogger(__name__)


class DreamBoothDataset(Dataset):
    """Custom dataset for DreamBooth fine-tuning."""

    def __init__(self, instance_dir: str, instance_prompt: str,
                 size: int = 512, center_crop: bool = True):
        self.instance_dir = Path(instance_dir)
        self.instance_prompt = instance_prompt
        self.image_paths = list(self.instance_dir.glob("*.png")) + \
                           list(self.instance_dir.glob("*.jpg")) + \
                           list(self.instance_dir.glob("*.jpeg"))

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        logger.info(f"Loaded {len(self.image_paths)} images from {instance_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "prompt": self.instance_prompt,
        }


class DreamBoothLoRATrainer:
    """Fine-tune Stable Diffusion with DreamBooth + LoRA."""

    def __init__(self, model_id: str, instance_prompt: str,
                 lora_rank: int = 16, lora_alpha: int = 32):
        self.model_id = model_id
        self.instance_prompt = instance_prompt

        logger.info(f"Loading model: {model_id}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None,
        ).to("cuda")

        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_v", "to_k", "to_out.0"],
            lora_dropout=0.05,
        )

    def prepare_model(self):
        self.pipe.unet = get_peft_model(self.pipe.unet, self.lora_config)
        self.pipe.unet.train()

        total = sum(p.numel() for p in self.pipe.unet.parameters())
        trainable = sum(p.numel() for p in self.pipe.unet.parameters()
                        if p.requires_grad)
        logger.info(f"Total params: {total:,} | Trainable: {trainable:,} "
                     f"({100 * trainable / total:.2f}%)")
        return trainable

    def compute_loss(self, pixel_values: torch.Tensor,
                     encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        latents = self.pipe.vae.encode(
            pixel_values.to(dtype=torch.float16)
        ).latent_dist.sample() * self.pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = self.pipe.unet(
            noisy_latents, timesteps, encoder_hidden_states,
        ).sample

        target = noise  # epsilon prediction
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    def train(self, dataset: DreamBoothDataset, epochs: int = 100,
              learning_rate: float = 1e-4, batch_size: int = 1,
              save_every: int = 25, output_dir: str = "./dreambooth_output"):
        self.prepare_model()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.pipe.unet.parameters(), lr=learning_rate, weight_decay=1e-2,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=50,
            num_training_steps=epochs * len(dataloader),
        )

        wandb.init(project="dreambooth-lora", config={
            "model": self.model_id, "lora_rank": self.lora_config.r,
            "epochs": epochs, "lr": learning_rate,
            "instance_prompt": self.instance_prompt,
        })

        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to("cuda")

                text_inputs = self.pipe.tokenizer(
                    batch["prompt"], padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                ).to("cuda")
                encoder_hidden_states = self.pipe.text_encoder(
                    text_inputs.input_ids
                )[0]

                loss = self.compute_loss(pixel_values, encoder_hidden_states)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.pipe.unet.parameters(), max_norm=1.0
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                wandb.log({
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "global_step": global_step,
                })

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

            if (epoch + 1) % save_every == 0:
                ckpt_path = output_path / f"checkpoint-{epoch + 1}"
                self.pipe.unet.save_pretrained(str(ckpt_path))
                logger.info(f"Saved checkpoint: {ckpt_path}")

        # Save final model
        final_path = output_path / "final"
        self.pipe.unet.save_pretrained(str(final_path))
        wandb.finish()
        logger.info(f"Training complete. Model saved to {final_path}")
        return final_path

    @torch.no_grad()
    def generate(self, prompt: str, num_images: int = 4,
                 guidance_scale: float = 7.5, num_steps: int = 50):
        self.pipe.unet.eval()
        images = self.pipe(
            prompt, num_images_per_prompt=num_images,
            guidance_scale=guidance_scale, num_inference_steps=num_steps,
        ).images
        return images


if __name__ == "__main__":
    trainer = DreamBoothLoRATrainer(
        model_id="stabilityai/stable-diffusion-2-1",
        instance_prompt="a photo of sks dog",
    )
    dataset = DreamBoothDataset(
        instance_dir="./data/my_dog",
        instance_prompt="a photo of sks dog",
    )
    trainer.train(dataset, epochs=100, learning_rate=1e-4)
    images = trainer.generate("a photo of sks dog in a park")
    for i, img in enumerate(images):
        img.save(f"output_{i}.png")
