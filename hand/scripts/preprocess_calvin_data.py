"""
Script to preprocess CALVIN data with 2D flow and embeddings.

Usage:
    uv run hand/scripts/preprocess_calvin_data.py \
        env_name=robot \
        dataset_name=playdata0 \
        compute_2d_flow=True \
        flow.text_prompt="robot. objects." \
        debug=True
"""
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from hydra import compose, initialize
from omegaconf import DictConfig

from calvin.calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv
from hand.retrieval.utils import get_tracked_points
from hand.data.optical_flow import load_cotracker
from hand.data.preprocess import compute_image_embeddings
from hand.data.utils import load_data_compressed, save_data_compressed
from hand.models.image_embedder import ImageEmbedder
from hand.utils.logger import log


def preprocess_calvin_data(cfg: DictConfig, data_dir: Path):
    """
    Assumes CALVIN data is stored in the following format:

        data_dir/
            ann_000000.dat

    Which stores data as:
        state_dict = [
            frame_ids,
            rgb_static,
            rgb_gripper,
            actions,
            rel_actions,
            joint_actions,
            proprios,
            scene_obs
        ]

    Creates a new directory in data_dir/processed_trajs/ with the following format
    with .dat files for each type of information:

        data_dir/
            processed_trajs/
                traj_000000/
                    depth_images.dat
                    external_images.dat
    """
    trajs = sorted(data_dir.glob("*.dat"))
    log(f"Processing {len(trajs)} trajectories", "yellow")

    if cfg.debug:
        trajs = trajs[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize necessary models!
    cotracker = load_cotracker(cfg.flow.cotracker_ckpt_file)
    cotracker = cotracker.to(device)

    # Make different image embedders
    image_embedders = {}
    image_embedders["dinov2_vitb14"] = ImageEmbedder(
        model_name="dinov2_vitb14", device=device
    )

    for embed_type in ["resnet18", "resnet50"]:
        for feature_map_layer in ["layer4", "avgpool"]:
            image_embedders[f"{embed_type}_{feature_map_layer}"] = ImageEmbedder(
                model_name=embed_type,
                device=device,
                feature_map_layer=feature_map_layer,
            )

    for embed_type in image_embedders:
        image_embedders[embed_type] = image_embedders[embed_type].to(device)

    for traj_idx, traj in enumerate(
        tqdm.tqdm(trajs, desc="Processing trajectories")
    ):
        # Save to .dat format
        new_traj_dir = data_dir / "processed_trajs" / f"traj_{traj_idx:06d}"
        new_traj_dir.mkdir(parents=True, exist_ok=True)
        save_file = new_traj_dir / "traj_data.dat"

        state_dict = load_data_compressed(traj)

        if not save_file.exists():
            # Save metadata for each trajectory
            traj_data = {
                "states": state_dict[6],
                "actions": state_dict[3],
                "rewards": np.zeros(len(state_dict[3])),
            }
            save_data_compressed(save_file, traj_data)

        # Initialize storage for images and embeddings
        camera_imgs = {}

        camera_types = ["external", "wrist"]
        for camera_type in camera_types:
            if camera_type == "external":
                images = state_dict[1]
                camera_imgs["external"] = images

            elif camera_type == "wrist":
                images = state_dict[2]
                camera_imgs["wrist"] = images

        # Save processed images
        for camera_type, images in camera_imgs.items():

            img_file = new_traj_dir / f"{camera_type}_images.dat"
            if not img_file.exists():
                save_data_compressed(img_file, camera_imgs[camera_type])

            # Process all embedding types that we want to save
            embedding_model = "dinov2_vitb14"

            img_embed_file = (
                new_traj_dir / f"{camera_type}_img_embeds_{embedding_model}.dat"
            )
            if not img_embed_file.exists():
                img_embeds = compute_image_embeddings(
                    embedder=image_embedders["dinov2_vitb14"], images=[images]
                )[0]
                save_data_compressed(img_embed_file, img_embeds)

            resnet_embedding_models = ["resnet18", "resnet50"]
            resnet_feature_map_layers = ["layer4", "avgpool"]

            for resnet_embedding_model in resnet_embedding_models:
                for resnet_feature_map_layer in resnet_feature_map_layers:
                    img_embed_file = (
                        new_traj_dir
                        / f"{camera_type}_img_embeds_{resnet_embedding_model}_{resnet_feature_map_layer}.dat"
                    )
                    if not img_embed_file.exists() and camera_type != "depth":
                        img_embeds = compute_image_embeddings(
                            embedder=image_embedders[
                                f"{resnet_embedding_model}_{resnet_feature_map_layer}"
                            ],
                            images=[images],
                        )[0]
                        save_data_compressed(img_embed_file, img_embeds)

        # Compute flow information and perform SAM 2 point tracking
        point_tracking_file = new_traj_dir / "2d_flow.dat"

        if not point_tracking_file.exists():
            video = camera_imgs["external"]

            GlobalHydra.instance().clear()
            with initialize(config_path="../../calvin/calvin_env/conf"):
                env_cfg = compose(config_name="config_data_collection",
                                  overrides=["cameras=static_and_gripper"])

                env_cfg.env["use_egl"] = False
                env_cfg.env["show_gui"] = False
                env_cfg.env["use_vr"] = False
                env_cfg.env["use_scene_info"] = True

                env_cfg.cameras.static.width = 200
                env_cfg.cameras.static.height = 200
                env_cfg = {**env_cfg.env}

                env_cfg.pop("_target_", None)
                env_cfg.pop("_recursive_", None)
                env = PlayTableSimEnv(**env_cfg)

            eef = state_dict[6][:, :3]
            tracked_points = get_tracked_points(video, eef, env, cotracker, cfg)
            save_data_compressed(point_tracking_file, tracked_points)


@hydra.main(version_base=None, config_name="convert_to_tfds", config_path="../cfg")
def main(cfg):
    """Main function to convert replay buffer to TFDS format."""
    data_dir = Path(cfg.data_dir)
    log(f"Processing data from {data_dir}", "yellow")
    preprocess_calvin_data(cfg, data_dir)


if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()
