"""
Retrieves K subtrajectories from play data from N queries from query_task
and converts retrieved trajectories to TFDS format.

Retrieval methods:
- STRAP: S-DTW w/ DINOv2 embeddings
- 3D: S-DTW w/ end-effector positions
- 2D: S-DTW w/ relative 2D paths from CoTracker
- HAND: Visual filtering w/ DINOV2 embeddings + relative 2D differences

Usage:

uv run hand/retrieval/retrieval_calvin.py \
    query_task=move_slider_left \
    query_source=expert \
    method=hand \
    K=250 \
    with_expert=True \
    save_dataset=True

uv run hand/retrieval/retrieval_calvin.py \
    query_task=move_slider_left \
    query_source=expert \
    method=2d \
    K=5000 \
    K2=100 \
    with_expert=True \
    save_dataset=True
    
"""

import pickle as pkl
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

from hand.retrieval.data_utils import format_to_tfds, get_data_paths
from hand.retrieval.dtw import get_single_match
from hand.retrieval.utils import (
    get_features,
    minimax_ln_scaling,
    visual_filtering,
)
from hand.retrieval.visualize import visualize_paths, visualize_retrieved_trajs
from hand.data.utils import load_data_compressed, save_dataset
from hand.resolvers import *  # noqa: F403
from hand.utils.general_utils import omegaconf_to_dict, print_dict
from hand.utils.logger import log

METHODS = ["strap", "3d", "2d_abs", "2d", "hand_abs", "hand"]


def retrieval(
    cfg: DictConfig,
    query_paths: List[Path],
    play_paths: List[Path],
    method: str,
    wandb_run: Optional = None,
    K: int = 100,
):
    N = len(query_paths)
    matches_per_query = K // N
    remainder = K % N

    log(f"Retrieving {matches_per_query} matches per query", color="cyan")

    # Retrieval
    all_retrieved_trajs = []
    for query_i in tqdm.tqdm(range(len(query_paths)), total=len(query_paths)):
        query_path = query_paths[query_i]

        matches = {}
        if cfg.query_source == "expert":  # expert returns data paths
            traj_data_file = query_path / "traj_data.dat"
            video_data_file = query_path / "external_images.dat"

            if traj_data_file.exists():
                traj_data = load_data_compressed(traj_data_file)
                T = len(traj_data["actions"])
            elif video_data_file.exists():
                video_data = load_data_compressed(video_data_file)
                T = video_data.shape[0]
            else:
                raise ValueError(
                    f"Missing traj_data.dat or external_images.dat for {query_path}"
                )

            # Split query trajectory in half (this is for CALVIN for now)
            start = 0 if query_i % 2 == 0 else T // 2
            end = T // 2 if query_i % 2 == 0 else -1

            if cfg.with_expert:
                # add query trajectories to matches
                matches[query_path] = (0, start, end)

            # get query feature
            query = get_features(
                traj_dir=query_path,
                method=method,
            )

            query = query[start:end]

        elif (
            cfg.query_source == "lang"
        ):
            query = query_path

        # Find matches in play data for each query
        for j, play_path in tqdm.tqdm(
            enumerate(play_paths),
            total=len(play_paths),
            desc=f"Query {query_i + 1}/{len(query_paths)}",
        ):
            if cfg.debug and j > 30:
                break

            play = get_features(
                traj_dir=play_path,
                method=method,
                env=cfg.env.env_name,
            )

            # S-DTW requires the reference trajectory to be longer than the query trajectory
            if len(play) < len(query):
                log(f"play: {len(play)} < query {len(query)}")
                continue

            # Get match info
            if cfg.query_source != "lang":
                info = get_single_match(query=query, play=play)
            else:
                similarity = np.dot(query, play) / (
                    np.linalg.norm(query) * np.linalg.norm(play)
                )
                info = (1 - similarity, 0, -1)

            # This happens for exact matches i.e., query matches play
            if np.isnan(info[0]):
                log(
                    f"Found NaN cost, skipping: {play_path}: start={info[1]}, end={info[2]}",
                    color="yellow",
                )
                continue

            matches[play_path] = info

        #  Sort by ascending cost
        retrieved_trajs = sorted(matches.items(), key=lambda item: item[1][0])

        if query_i == 0 and remainder > 0:
            retrieved_trajs = retrieved_trajs[: matches_per_query + remainder]
        else:
            retrieved_trajs = retrieved_trajs[:matches_per_query]

        # Visualize retrieved trajectories
        all_videos, all_frames = visualize_retrieved_trajs(cfg, retrieved_trajs)

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"retrieved_subtrajs_{query_i}": all_videos,
                    f"retrieved_subtrajs_{query_i}/keyframes": all_frames,
                }
            )

        fig = visualize_paths(retrieved_trajs)

        if wandb_run is not None:
            wandb_run.log({"retrieval_stats/trajectory": [wandb.Image(fig)]})

        # use features as cost unless path cost weighting
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

    log(
        f"Processed {len(all_retrieved_trajs)} retrieved trajectories = K [{cfg.K}], demos: N [{N}]",
        color="red",
    )

    return all_retrieved_trajs


def plot_cost_distribution(costs, wandb_run):
    fig = plt.figure()
    plt.hist(costs, bins=50)
    plt.xlabel("Cost")
    plt.title("Cost distribution before normalization")
    plt.axvline(np.mean(costs), color="red", linestyle="--")
    if wandb_run is not None:
        wandb_run.log({"retrieval_stats/costs": [wandb.Image(fig)]})
    plt.close()


@hydra.main(version_base=None, config_name="retrieval_calvin", config_path="../cfg")
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

    # Check if valid
    if cfg.query_source == "expert":
        assert (
            cfg.method in METHODS
        ), f"Invalid method {cfg.method}. Supported methods: {METHODS}"
    elif cfg.query_source == "lang":
        raise ValueError("Language-based query source is not supported. Use expert.")
    else:
        raise ValueError(
            f"Invalid query source {cfg.query_source}. Supported sources: expert, hamster"
        )

    log("-" * 50, color="cyan")
    log("Retrieval configs:", color="cyan")
    log(f"method: {cfg.method}", color="cyan")
    log(f"with_expert: {cfg.with_expert}", color="cyan")
    log(f"save_dataset: {cfg.save_dataset}", color="cyan")
    log(f"K: {cfg.K}", color="cyan")
    log(f"N: {cfg.N}", color="cyan")
    log("-" * 50, color="cyan")

    # Set up data paths for CALVIN
    log("Setting up data paths for retrieval", color="yellow")

    # log retrieval information
    log("-" * 50, color="cyan")
    log(f"query_task: {cfg.query_task}", color="cyan")
    log(f"query_env: {cfg.query_env}", color="cyan")
    log(f"play_envs: {cfg.play_envs}", color="cyan")
    log(f"query_source: {cfg.query_source}", color="cyan")
    log("-" * 50, color="cyan")

    query_paths, play_paths = get_data_paths(cfg=cfg)

    if cfg.query_source == "expert":
        # Visualize query videos
        for idx, query_path in tqdm.tqdm(
            enumerate(query_paths),
            total=len(query_paths),
            desc="Visualizing query videos",
        ):
            video_file = query_path / "external_images.dat"
            video = load_data_compressed(video_file)

            if wandb_run is not None:
                video = video.transpose(0, 3, 1, 2)
                wandb_run.log(
                    {
                        "query_video": [
                            wandb.Video(
                                video, caption="Query Video", fps=30, format="mp4"
                            )
                        ]
                    }
                )

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

        K = cfg.K2  # use K2 for second step retrieval

    # retrieve subtrajectories
    all_retrieved_trajs = retrieval(
        cfg=cfg,
        query_paths=query_paths,
        play_paths=play_paths,
        method=cfg.method,
        wandb_run=wandb_run,
        K=K,
    )

    # save metadata
    output_dir = Path(cfg.paths.root_dir).parent / "retrieval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / f"{cfg.query_task}_{cfg.method}_retrieval_metadata.pkl", "wb"
    ) as f:
        pkl.dump(all_retrieved_trajs, f)

    log(
        f"Saved metadata to {output_dir / f'{cfg.query_task}_{cfg.method}_retrieval_metadata.pkl'}",
        color="yellow",
    )

    log(
        f"Processed {len(all_retrieved_trajs)} retrieved trajectories = K [{K}], demos: N [{cfg.N}]",
        color="red",
    )

    trajectories = format_to_tfds(cfg, all_retrieved_trajs)

    # Save dataset
    if cfg.save_dataset:
        task_name = cfg.query_task.replace("_", "-")
        folder_name = "retrieval"

        # add some info to the folder name
        if cfg.with_expert:
            folder_name += "_with_expert"
        if "hand" in cfg.method:
            folder_name += "_two_step"

        save_file = (
            Path(cfg.paths.data_dir)
            / "tensorflow_datasets"
            / folder_name
            / cfg.dataset_name
            / cfg.method
            / f"{task_name}_N-{cfg.N}_K-{cfg.K}"
        )
        save_dataset(trajectories, save_file=save_file)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
