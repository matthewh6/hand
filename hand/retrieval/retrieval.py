"""
Retrieves K subtrajectories from play data from N queries from query_task
and converts retrieved trajectories to TFDS format.

Retrieval methods:
- STRAP: S-DTW w/ DINOV2 embeddings
- 3D: S-DTW w/ end-effector positions
- 2D: S-DTW w/ 2D paths from CoTracker
- HAND: Visual filtering w/ DINOV2 embeddings and 2D S-DTW

Usage:
    
uv run hand/retrieval/retrieval.py \
    query_task=close_microwave \
    query_source=expert \
    method=hand \
    K=250 \
    K2=25 \
    with_expert=True \
    save_dataset=True

"""

import random
from pathlib import Path
from typing import List, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import DictConfig

from hand.retrieval.dtw import get_single_match
from hand.retrieval.utils import (
    get_features,
    minimax_ln_scaling,
    visual_filtering,
)
from hand.retrieval.visualize import (
    add_flow_to_video,
    visualize_retrieved_trajs,
)
from hand.data.utils import load_data_compressed, raw_data_to_tfds
from hand.resolvers import *  # noqa: F403
from hand.utils.general_utils import omegaconf_to_dict, print_dict
from hand.utils.logger import log

METHODS = ["strap", "3d", "2d_abs", "2d", "hand_abs", "hand"]


def plot_cost_distribution(costs, wandb_run):
    fig = plt.figure()
    plt.hist(costs, bins=50)
    plt.xlabel("Cost")
    plt.title("Cost distribution before normalization")
    plt.axvline(np.mean(costs), color="red", linestyle="--")
    if wandb_run is not None:
        wandb_run.log({"retrieval_stats/costs": [wandb.Image(fig)]})
    plt.close()


def retrieval(
    cfg: DictConfig,
    query_paths: List[Path],
    play_paths: List[Path],
    method: str,
    wandb_run: Optional = None,
    K: int = 100,
):
    matches_per_query = K // cfg.N
    remainder = K % cfg.N
    log(f"Retrieving {matches_per_query} matches per query", color="cyan")

    all_retrieved_trajs = []
    for query_i in tqdm.tqdm(range(len(query_paths)), total=len(query_paths)):
        query_traj_dir = query_paths[query_i]

        matches = {}
        if cfg.query_source == "expert":  # expert returns data paths
            traj_data_file = query_traj_dir / "traj_data.dat"
            video_data_file = query_traj_dir / "external_images.dat"
            if traj_data_file.exists():
                traj_data = load_data_compressed(traj_data_file)
                T = len(traj_data["actions"])
            elif video_data_file.exists():
                video_data = load_data_compressed(video_data_file)
                T = video_data.shape[0]
            else:
                raise ValueError(
                    f"Missing traj_data.dat or external_images.dat for {query_traj_dir}"
                )

            if cfg.with_expert:
                # add expert trajectory to matches
                matches[query_traj_dir] = (0, 0, T)

            # get query feature based on method
            query_feature = get_features(
                traj_dir=query_traj_dir,
                method=method,
            )

        # Find matches in play data for each query
        for j, other_traj_dir in tqdm.tqdm(
            enumerate(play_paths),
            total=len(play_paths),
            desc=f"Query {query_i + 1}/{len(query_paths)}",
        ):
            if cfg.debug and j > 30:
                break

            other_feature = get_features(
                traj_dir=other_traj_dir,
                method=method,
            )

            # S-DTW requires the reference trajectory to be longer than the query trajectory
            if len(other_feature) < len(query_feature):
                log(f"other: {len(other_feature)} < query {len(query_feature)}")
                continue

            # Get match info
            info = get_single_match(query=query_feature, play=other_feature)

            # This happens for exact matches i.e., query matches play
            if np.isnan(info[0]):
                log(
                    f"Found NaN cost, skipping: {other_traj_dir}: start={info[1]}, end={info[2]}",
                    color="yellow",
                )
                continue

            matches[other_traj_dir] = info

        # Sort by ascending cost
        retrieved_trajs = sorted(matches.items(), key=lambda item: item[1][0])

        if query_i == 0 and remainder > 0:
            retrieved_trajs = retrieved_trajs[: matches_per_query + remainder]
        else:
            retrieved_trajs = retrieved_trajs[:matches_per_query]

        # Visualize retrieved trajectories
        if wandb_run is not None:
            all_videos, all_frames = visualize_retrieved_trajs(cfg, retrieved_trajs)
            wandb_run.log(
                {
                    f"retrieved_subtrajs_{query_i}_{method}": all_videos,
                    f"retrieved_subtrajs_{query_i}_{method}/keyframes": all_frames,
                }
            )

        costs = np.array([-metrics[0] for _, metrics in retrieved_trajs])
        if cfg.with_expert:
            if len(costs) > 1:
                costs[0] = costs[1]  # since query has C=0

        # Plot cost map before cost and after cost normalization
        plot_cost_distribution(costs, wandb_run)

        # Normalize costs for weighted BC
        costs = minimax_ln_scaling(costs)
        costs = np.exp(costs)

        # update retrieval matches with normalized costs
        for i, (play_path, info) in enumerate(retrieved_trajs):
            retrieved_trajs[i] = (play_path, (costs[i], info[1], info[2]))

        # Process matches and get retrieved trajectories
        all_retrieved_trajs.extend(retrieved_trajs)

    return all_retrieved_trajs


@hydra.main(version_base=None, config_name="retrieval", config_path="../cfg")
def main(cfg: DictConfig):
    print_dict(omegaconf_to_dict(cfg))

    # Set some random seeds
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # Initialize wandb if enabled
    if cfg.use_wandb:
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project="hand-demos-retrieval",
            name=cfg.wandb.name,
            config=omegaconf_to_dict(cfg),
        )
    else:
        wandb_run = None

    # Check if valid retrieval method
    assert (
        cfg.method in METHODS
    ), f"Invalid method {cfg.method}. Supported methods: {METHODS}"

    log("-" * 50, color="cyan")
    log("Retrieval configs:", color="cyan")
    log(f"method: {cfg.method}", color="cyan")
    log(f"with_expert: {cfg.with_expert}", color="cyan")
    log(f"K: {cfg.K}", color="cyan")
    log(f"N: {cfg.N}", color="cyan")
    log("-" * 50, color="cyan")

    # Set up data paths for CALVIN
    log("Setting up data paths for retrieval", color="yellow")

    # log retrieval information
    log("-" * 50, color="cyan")
    log(f"query_task: {cfg.query_task}", color="cyan")
    log(f"other_tasks: {cfg.other_tasks}", color="cyan")
    log(f"query_source: {cfg.query_source}", color="cyan")
    log("-" * 50, color="cyan")

    if cfg.query_files is not None:
        query_paths = [Path(query_file) for query_file in cfg.query_files]
    else:
        query_dir = (
            Path(cfg.paths.data_dir) / "datasets" / cfg.dataset_name / cfg.query_task
        )
        query_paths = list((query_dir / "subtraj_data").glob("subtraj_*"))
        query_indices = np.random.choice(
            range(len(query_paths)), size=cfg.N, replace=False
        )
        query_paths = [query_paths[i] for i in query_indices]

    play_paths = []
    for other_task in cfg.other_tasks:
        other_dir = (
            Path(cfg.paths.data_dir) / "datasets" / cfg.dataset_name / other_task
        )

        play_paths.extend(list((other_dir / "subtraj_data").glob("subtraj_*")))

    play_paths = sorted(play_paths)

    # Visualize query videos
    all_query_videos = []
    if wandb_run is not None:
        for idx, query_traj_dir in tqdm.tqdm(
            enumerate(query_paths),
            total=len(query_paths),
            desc="Visualizing query videos",
        ):
            video_file = query_traj_dir / "external_images.dat"
            video = load_data_compressed(video_file)

            if "2d" in cfg.method:
                flow_file = query_traj_dir / "2d_flow_query.dat"
                flow = load_data_compressed(flow_file)["points"][:, 0]
                video = add_flow_to_video(video, flow)

            video = video.transpose(0, 3, 1, 2)
            all_query_videos.append(
                wandb.Video(video, caption=f"Query Video_{idx}", fps=30, format="mp4")
            )

        wandb_run.log({"query_videos": all_query_videos})

    K = cfg.K

    # Retrieval
    if "hand" in cfg.method:
        log("Running visual filtering step", color="yellow")

        # First retrieve with dino_fmi features
        all_retrieved_trajs = visual_filtering(
            cfg=cfg,
            query_paths=query_paths,
            play_paths=play_paths,
            method="dino_fmi",
            wandb_run=wandb_run,
            K=K,
        )

        play_paths = all_retrieved_trajs
        log(f"Filtered to {len(all_retrieved_trajs)} trajectories", color="yellow")

        K = cfg.K2  # K2 for second step retrieval

    # Retrieve subtrajectories
    all_retrieved_trajs = retrieval(
        cfg,
        query_paths,
        play_paths,
        cfg.method,
        wandb_run,
        K=K,
    )

    # Save dataset
    if cfg.save_dataset:
        tag = (
            f"{cfg.query_task}_retrieval_N-{cfg.N}_K-{cfg.K}_f-{cfg.method}"
            if "hand" not in cfg.method
            else f"{cfg.query_task}_retrieval_N-{cfg.N}_K-{cfg.K}_K2-{cfg.K2}_f-{cfg.method}"
        )

        save_file = (
            Path(cfg.paths.data_dir) / "tensorflow_datasets" / cfg.env.env_name / tag
        )

        traj_dirs, segments, costs = zip(
            *[
                (traj_dir, [start, end], cost)
                for traj_dir, (cost, start, end) in all_retrieved_trajs
            ]
        )

        raw_data_to_tfds(
            traj_dirs,
            save_file,
            embedding_model=cfg.embedding_model,
            resnet_feature_map_layer=cfg.resnet_feature_map_layer,
            segments=segments,
            costs=costs,
        )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
