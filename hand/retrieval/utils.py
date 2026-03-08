from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.nn import functional as F

from hand.data.optical_flow import generate_point_tracks
from hand.data.utils import load_data_compressed
from hand.retrieval.visualize import visualize_retrieved_trajs
from hand.resolvers import *  # noqa: F403
from hand.utils.general_utils import to_numpy
from hand.utils.logger import log


TASK_TO_LANG = {
    "move_slider_left": [
        "move the robot arm to the slider",
        "move the slider to the left",
    ],
    "open_drawer": ["move the robot arm to the drawer", "open the drawer"],
    "close_drawer": ["move the robot arm to the drawer", "close the drawer"],
    "lift_blue_block_table": [
        "move the robot arm to the blue block",
        "lift the blue block",
    ],
    "turn_on_led": [
        "move the robot arm to the button",
        "push down on the button",
    ],
}

index_map = {
    "frame_ids": 0,
    "rgb_static": 1,
    "rgb_gripper": 2,
    "actions": 3,
    "rel_actions": 4,
    "joint_actions": 5,
    "proprios": 6,
    "scene_obs": 7,
    "flow": 8,  # 2D point tracks (CoTracker) for HAND
    "dino_embs": 9,
    "rgb_static_embs": 10,
    "rgb_gripper_embs": 11,
}


def get_state_dict_features(state_dict: list, method: str):
    """
    Returns the corresponding features for the given query_type.

    Args:
        state_dict (list): list of features
        method (str): method to use for retrieval
    Returns:
        query (np.ndarray): features for the given query_type
    """

    if method == "strap":
        features = state_dict[index_map["dino_embs"]]

    elif method == "3d":
        features = state_dict[index_map["rel_actions"]][:, :3]  # 3d relative actions

    elif "2d" in method or "hand" in method:
        if isinstance(state_dict[8], dict):
            features = state_dict[8]["points"].reshape(-1, 2)
        else:
            features = state_dict[8].reshape(-1, 2)

        if "abs" not in method:
            # Compute relative 2d flow
            features = np.diff(features, axis=0)

    elif method == "dino_fmi":
        features = state_dict[index_map["dino_embs"]]
        start_emb = features[0]
        end_emb = features[-1]

        return {
            "start": torch.tensor(start_emb, dtype=torch.float32),
            "end": torch.tensor(end_emb, dtype=torch.float32),
        }

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.array(features)


def visual_filtering(cfg, query_paths, play_paths, method, wandb_run, K: int = None):
    log(
        f"Visual-filtering with K: {K} and method: {method}",
        color="yellow",
    )

    matches_per_query = K // cfg.N
    remainder = K % cfg.N
    log(f"Retrieving {matches_per_query} matches per query", color="cyan")

    # Precompute features for all other trajectories
    all_other_feats = []
    for i, traj_dir in tqdm.tqdm(
        enumerate(play_paths),
        total=len(play_paths),
        desc="Loading other features",
    ):
        if cfg.debug and i > 30:
            break

        feats = get_features(traj_dir=traj_dir, method=method)

        all_other_feats.append(feats)

    all_retrieved = []

    for query_i, query_dir in enumerate(query_paths):
        query_feats = get_features(
            traj_dir=query_dir, method=method
        )

        scores = []
        for j, other_feats in enumerate(all_other_feats):
            if cfg.debug and j > 30:
                break

            score = F.mse_loss(
                query_feats["start"], other_feats["start"]
            ) + F.mse_loss(query_feats["end"], other_feats["end"])

            scores.append((score.item(), j))

        # sort by S-DTW cost
        scores.sort(key=lambda x: x[0])

        # get top K matches
        num_matches = matches_per_query + (remainder if query_i == 0 else 0)
        top_scores = scores[:num_matches]

        # store retrieved trajectories
        retrieved = [play_paths[j] for _, j in top_scores]
        all_retrieved.extend(retrieved)

        if wandb_run is not None:
            # use the ordered top_scores for visualization
            viz_matches = [(play_paths[j], (score, 0, -1)) for score, j in top_scores]
            vids, frames = visualize_retrieved_trajs(cfg, viz_matches)
            wandb_run.log(
                {
                    f"retrieved_subtrajs_{query_i}_dino_fmi": vids,
                    f"retrieved_subtrajs_{query_i}_dino_fmi/keyframes": frames,
                }
            )

    return all_retrieved


def get_features(
    traj_dir: Path,
    method: str,
):
    if method == "strap":
        embed_dat_file = traj_dir / "external_img_embeds_dinov2_vitb14.dat"
        features = load_data_compressed(embed_dat_file)

    elif method == "3d":
        traj_data_file = traj_dir / "traj_data.dat"
        traj_data = load_data_compressed(traj_data_file)
        features = traj_data["states"][:, :3]

    elif "2d" in method or "hand" in method:
        flow_file = traj_dir / "2d_flow_query.dat"
        flow_data = load_data_compressed(flow_file)
        normalized_flow_data = flow_data["points_normalized"]
        features = normalized_flow_data[:, 0]

        if "abs" not in method:
            features = features[:, 1:] - features[:, :-1]

    elif method == "dino_fmi":
        embed_dat_file = traj_dir / "external_img_embeds_dinov2_vitb14.dat"
        features = load_data_compressed(embed_dat_file)

        start_emb = features[0]
        end_emb = features[-1]

        features = {
            "start": torch.tensor(start_emb, dtype=torch.float32),
            "end": torch.tensor(end_emb, dtype=torch.float32),
        }

    else:
        raise ValueError(f"Unknown method: {method}")

    return features


def get_tracked_points(video, eef, env, cotracker, cfg):
    # figure out where the eef is in the image space
    eef_hom = np.concatenate([eef, np.ones((eef.shape[0], 1))], axis=1)
    xy = env.cameras[0].project(eef_hom[0])
    query = np.array([0, xy[0], xy[1] - 20])
    queries = np.array([query])

    points, _ = generate_point_tracks(cotracker=cotracker, video=video, queries=queries)

    return to_numpy(points[0]).squeeze()


def minimax_ln_scaling(val):
    if len(val) <= 1 or np.max(val) == np.min(val):
        return np.full_like(val, np.log(100))

    val = (val - np.min(val)) / (np.max(val) - np.min(val))  # Normalize to [0, 1]
    log_min, log_max = np.log(1e-2), np.log(100)  # [-4.605, 4.605]
    return val * (log_max - log_min) + log_min
