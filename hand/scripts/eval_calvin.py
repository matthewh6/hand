"""
Evaluate a trained ACT policy on the CALVIN benchmark.

Wraps the trained SimpleACTPolicy in the CalvinBaseModel interface and runs
the standard CALVIN multi-step evaluation (1000 sequences of 5 chained tasks).

Requires: pip install pytorch-lightning pybullet

Usage:
    uv run hand/scripts/eval_calvin.py \
        --checkpoint results/checkpoints/move-slider-left_N-6_K-250/best.pt \
        --dataset_path calvin/dataset/task_D_D
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from hand.scripts.train_act import SimpleACTPolicy
from hand.utils.logger import log


def load_act_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained SimpleACTPolicy from checkpoint."""
    device = torch.device(device)

    log(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(ckpt["cfg"])
    model = SimpleACTPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log(f"Loaded model from epoch {ckpt['epoch']} (loss: {ckpt['loss']:.6f})")
    return model, device


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy on CALVIN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="calvin/dataset/task_D_D",
        help="Path to CALVIN dataset (contains validation/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show visualization during evaluation",
    )
    args = parser.parse_args()

    # Import CALVIN deps (requires pytorch-lightning, pybullet)
    calvin_root = Path(__file__).resolve().parents[2] / "calvin"
    sys.path.insert(0, str(calvin_root / "calvin_models"))
    sys.path.insert(0, str(calvin_root / "calvin_env"))

    from calvin_agent.evaluation.evaluate_policy import evaluate_policy
    from calvin_agent.evaluation.utils import count_success
    from calvin_agent.models.calvin_base_model import CalvinBaseModel
    from calvin_env.envs.play_table_env import get_env

    # Load model
    model_nn, device = load_act_model(args.checkpoint, args.device)

    class ACTCalvinModel(CalvinBaseModel):
        """Wraps a trained SimpleACTPolicy for CALVIN evaluation."""

        def reset(self):
            pass

        def step(self, obs, goal):
            if "rgb_obs" in obs:
                image = obs["rgb_obs"]["rgb_static"]
            else:
                image = obs["rgb_static"]

            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image_t = torch.from_numpy(image).unsqueeze(0).to(device)

            state = obs["robot_obs"].astype(np.float32)
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)

            action = model_nn.predict_action(image_t, state_t)
            return action.cpu().numpy().squeeze()

    log("Creating CALVIN environment...", "yellow")
    val_folder = Path(args.dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    calvin_model = ACTCalvinModel()

    log("Starting CALVIN evaluation (1000 sequences x 5 tasks)...", "green")
    results = evaluate_policy(
        calvin_model,
        env,
        epoch=0,
        eval_log_dir=str(Path(args.checkpoint).parent / "eval_results"),
        debug=args.debug,
    )

    avg_len = np.mean(results)
    log(f"Average successful sequence length: {avg_len:.2f}", "green")

    for i, sr in enumerate(count_success(results)):
        log(f"  {i + 1} tasks in a row: {sr * 100:.1f}%", "cyan")


if __name__ == "__main__":
    main()
