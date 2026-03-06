"""CoTracker utilities for 2D point tracking."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hand.utils.logger import log


def load_cotracker(cotracker_ckpt_path: str):
    """Load CoTracker model from checkpoint."""
    from cotracker.predictor import CoTrackerPredictor

    log("Initializing CoTracker model")
    model = CoTrackerPredictor(checkpoint=cotracker_ckpt_path)
    return model


def generate_point_tracks(
    cotracker,
    video: np.ndarray,
    segm_mask: Optional[np.ndarray] = None,
    queries: Optional[np.ndarray] = None,
    grid_size: int = 25,
    device: str = "cuda",
):
    """
    Generate point tracks using CoTracker.

    Args:
        video: (T, H, W, 3) numpy array
        queries: (N, 3) numpy array - [frame_idx, x, y] for each query point
    """
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    video = video.to(device)

    log("Running CoTracker on the video")

    if queries is not None:
        log(f"Using provided queries: {queries}")
        queries = torch.from_numpy(queries).float().to(device)

    if queries is not None:
        pred_tracks, pred_visibility = cotracker(
            video, queries=queries[None], backward_tracking=True
        )
    else:
        pred_tracks, pred_visibility = cotracker(
            video,
            grid_size=grid_size,
            segm_mask=torch.from_numpy(segm_mask)[None, None],
        )
    log(
        f"Predicted tracks shape: {pred_tracks.shape}, visibility shape: {pred_visibility.shape}"
    )
    return pred_tracks, pred_visibility
