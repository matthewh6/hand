"""Preprocessing utilities."""

from typing import List

import numpy as np
import torch
import tqdm

from hand.utils.logger import log


def compute_image_embeddings(
    embedder,
    images: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute embeddings for a sequence of images using the specified embedder."""
    log(f"Computing image embeddings using {embedder.model_name}")
    embeddings = []

    for video in tqdm.tqdm(images, desc="computing embeddings"):
        with torch.no_grad():
            emb = embedder(np.array(video))
        embeddings.append(emb.cpu().numpy())

    return embeddings
