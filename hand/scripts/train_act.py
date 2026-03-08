"""
Train an ACT policy on a retrieved TFDS dataset for CALVIN.

Loads a TFDS dataset (saved by retrieval_calvin.py with save_dataset=True),
trains an Action Chunking Transformer policy, and saves checkpoints.

Usage:
    uv run hand/scripts/train_act.py \
        dataset_path=data/tensorflow_datasets/retrieval_with_expert_two_step/calvin/hand/move-slider-left_N-6_K-250
"""

import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from hand.utils.logger import log


class CalvinTFDSDataset(Dataset):
    """Load a TFDS dataset saved by retrieval_calvin.py."""

    def __init__(self, dataset_path: str, chunk_size: int = 20, image_size=(200, 200)):
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        self.chunk_size = chunk_size

        log(f"Loading dataset from {dataset_path}")
        ds = tf.data.Dataset.load(str(dataset_path))

        self.trajectories = []
        for traj in ds:
            traj_dict = {k: v.numpy() for k, v in traj.items()}
            if len(traj_dict["actions"]) >= chunk_size:
                self.trajectories.append(traj_dict)

        log(f"Loaded {len(self.trajectories)} trajectories (>= {chunk_size} steps)")

        self.samples = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["actions"])
            for t in range(T - chunk_size):
                self.samples.append((traj_idx, t))

        log(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_idx, t = self.samples[idx]
        traj = self.trajectories[traj_idx]

        image = traj["images"][t].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        state = traj["observations"][t].astype(np.float32)
        actions = traj["actions"][t : t + self.chunk_size].astype(np.float32)

        return {
            "image": torch.from_numpy(image),
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
        }


class SimpleACTPolicy(nn.Module):
    """Self-contained ACT policy for CALVIN.

    Architecture: frozen DINOv2 image encoder -> concat with state -> transformer decoder -> action chunks.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Image encoder (frozen DINOv2)
        self.image_encoder = torch.hub.load("facebookresearch/dinov2", cfg.image_encoder)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()

        img_embed_dim = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}[cfg.image_encoder]

        self.image_transforms = torch.nn.Sequential(
            # DINOv2 expects 224x224 normalized images
        )

        # Project image + state to d_model
        self.input_proj = nn.Linear(img_embed_dim + cfg.state_dim, cfg.d_model)

        # Transformer decoder for action chunking
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.num_layers
        )

        # Positional encoding for action queries
        self.action_pos_embed = nn.Embedding(cfg.chunk_size, cfg.d_model)

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.action_dim),
        )

        # Image normalization (DINOv2)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode_image(self, image):
        """Encode image with frozen DINOv2."""
        # Resize to 224x224 and normalize
        image = torch.nn.functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = (image - self.img_mean) / self.img_std

        with torch.no_grad():
            features = self.image_encoder(image)
        return features

    def forward(self, image, state):
        """Forward pass: image (B,3,H,W), state (B,state_dim) -> actions (B,chunk_size,action_dim)."""
        B = image.shape[0]

        # Encode image
        img_features = self.encode_image(image)  # (B, img_embed_dim)

        # Concatenate with state and project
        combined = torch.cat([img_features, state], dim=-1)
        memory = self.input_proj(combined).unsqueeze(1)  # (B, 1, d_model)

        # Action queries with positional encoding
        pos_ids = torch.arange(self.cfg.chunk_size, device=image.device)
        action_queries = self.action_pos_embed(pos_ids).unsqueeze(0).expand(B, -1, -1)

        # Decode
        output = self.transformer_decoder(tgt=action_queries, memory=memory)

        # Predict actions
        actions = self.action_head(output)  # (B, chunk_size, action_dim)
        return actions

    def predict_action(self, image, state):
        """Predict a single action (for evaluation)."""
        self.eval()
        with torch.no_grad():
            actions = self.forward(image, state)
        return actions[:, 0]  # Return first action in chunk


@hydra.main(version_base=None, config_name="train_act", config_path="../cfg")
def main(cfg: DictConfig):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = CalvinTFDSDataset(cfg.dataset_path, chunk_size=cfg.chunk_size)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model = SimpleACTPolicy(cfg).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {trainable_params:,} trainable / {total_params:,} total parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # Save directory
    save_dir = Path(cfg.paths.results_dir) / "checkpoints" / Path(cfg.dataset_path).name
    save_dir.mkdir(parents=True, exist_ok=True)
    log(f"Saving checkpoints to {save_dir}")

    # Training loop
    best_loss = float("inf")
    for epoch in range(cfg.num_epochs):
        model.train()
        model.image_encoder.eval()  # Keep image encoder frozen

        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            target_actions = batch["actions"].to(device)

            pred_actions = model(image, state)
            loss = nn.functional.mse_loss(pred_actions, target_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % cfg.eval_every == 0 or epoch == 0:
            log(
                f"Epoch {epoch + 1}/{cfg.num_epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}",
                "cyan",
            )

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = save_dir / f"epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "cfg": dict(cfg),
                },
                ckpt_path,
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "loss": avg_loss,
                    "cfg": dict(cfg),
                },
                save_dir / "best.pt",
            )

    log(f"Training complete. Best loss: {best_loss:.6f}", "green")
    log(f"Checkpoints saved to {save_dir}", "green")


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
