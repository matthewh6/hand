"""
Pipeline from collected data to training policies for baseline and our method.
"""

import itertools
import subprocess
import time
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from hand.utils.logger import log


def print_step_header(step_num: int, total_steps: int, desc: str):
    """Print a nicely formatted step header."""
    width = 80
    log("=" * width, color="blue")
    step_str = f"Step {step_num}/{total_steps}: {desc}"
    padding = (width - len(step_str)) // 2
    log(" " * padding + step_str, color="blue")
    log("=" * width, color="blue")


def run_command(
    cmd: str,
    desc: str = "",
    is_slurm: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run a command and log its output with timing."""
    if desc:
        log(f"📋 {desc}", color="cyan")
    log(f"$ {cmd}", color="yellow")

    if dry_run:
        return True

    start_time = time.time()
    try:
        if is_slurm:
            # For retrieval, use simple sbatch wrap
            subprocess.run(
                f"sbatch -c 8 --wrap='{cmd}'",
                shell=True,
                check=True,
            )
            log("✅ Job submitted to Slurm", color="green")
        else:
            subprocess.run(cmd, shell=True, check=True)
            duration = time.time() - start_time
            log(f"✅ Completed in {duration:.2f}s", color="green")
        return True
    except subprocess.CalledProcessError as e:
        log(f"❌ Failed with error code {e.returncode}", color="red")
        return False


def get_retrieved_dataset_name(
    dataset_name: str,
    N: int,
    K: int,
    feature_type: str,
    embedding_model: str,
    K2: int = None,
) -> str:
    """Construct the dataset name following the retrieval naming convention."""
    return (
        f"{dataset_name}_retrieval_N-{N}_K-{K}_f-{feature_type}"
        if "hand" not in feature_type
        else f"{dataset_name}_retrieval_N-{N}_K-{K}_K2-{K2}_f-{feature_type}"
    )


def generate_combinations(search_space: dict) -> List[dict]:
    """
    Generate all combinations from a search space dictionary.
    Example search_space: {
        'N': [1, 2],
        'K': [30, 50],
        'feature_type': ['2d', '3d']
    }
    """
    # Get all keys and their corresponding value lists
    keys = list(search_space.keys())
    value_lists = [search_space[k] for k in keys]

    # Generate all combinations
    combinations = list(itertools.product(*value_lists))

    # Convert to list of dictionaries
    return [dict(zip(keys, combo)) for combo in combinations]


def format_params(params: dict) -> str:
    """Format parameters for logging and run_id."""
    return "_".join(f"{k}-{v}" for k, v in params.items())


def update_command(base_cmd: str, params: dict) -> str:
    """Update command string with parameters."""
    param_strs = [f"{k}={v}" for k, v in params.items()]
    return base_cmd + " " + " ".join(param_strs)


def get_user_confirmation(message: str) -> bool:
    """Get user confirmation for an action."""
    response = input(f"\n{message} (y/n): ").lower().strip()
    return response == "y"


def check_processed_data(data_dir: str, env_name: str, task_name: str) -> bool:
    """Check if processed data exists for a task."""
    processed_path = (
        Path(data_dir) / "datasets" / env_name / task_name / "processed_trajs"
    )
    return processed_path.exists()


def check_subtraj_data(data_dir: str, env_name: str, task_name: str) -> bool:
    """Check if subtrajectory data exists for a task."""
    subtraj_path = Path(data_dir) / "datasets" / env_name / task_name / "subtraj_data"
    return subtraj_path.exists()


def preprocess_dataset(cfg: DictConfig, task_name: str, desc: str) -> bool:
    """Run preprocessing steps for a single dataset (split into subtrajectories only)."""
    log(f"\n📦 Processing {desc}", color="cyan")

    needs_preprocessing = not check_processed_data(
        cfg.paths.data_dir, cfg.env_name, task_name
    )
    needs_splitting = not check_subtraj_data(
        cfg.paths.data_dir, cfg.env_name, task_name
    )

    if needs_preprocessing:
        log(
            f"⚠️ Processed data not found for {task_name}. "
            "Ensure data is preprocessed (traj_data.dat, embeddings) before running retrieval.",
            color="yellow",
        )
        return False

    if not needs_splitting and not cfg.force_split:
        log(f"✅ {desc} already split", color="green")
        return True

    # Split into subtrajectories
    if not run_command(
        f"uv run hand/retrieval/scripts/split_data_into_subtrajs.py "
        f"dataset_name={cfg.env_name} "
        f"task={task_name}",
        f"Creating subtrajectory segments for {desc}",
    ):
        return False

    return True


def create_run_id(query_task: str, params: dict) -> str:
    """Create a unique run ID based on query task and parameters."""
    # Clean up query task name to be more concise
    task_name = query_task.replace("_", "-")

    # Format parameters
    param_str = format_params(params)

    # Combine task and parameters
    return f"{task_name}_{param_str}"


def check_retrieval_dataset(data_dir: str, dataset_name: str) -> bool:
    """Check if retrieval dataset exists."""

    tfds_path = Path(data_dir) / "tensorflow_datasets" / "robot" / dataset_name

    log(f"Checking for retrieval dataset: {tfds_path}", color="cyan")

    return tfds_path.exists()


def wait_for_dataset(data_dir: str, dataset_name: str, timeout: int = 3600) -> bool:
    """Wait for retrieval dataset to become available."""
    start_time = time.time()
    log(f"Waiting for retrieval dataset: {dataset_name}", color="cyan")

    while time.time() - start_time < timeout:
        if check_retrieval_dataset(data_dir, dataset_name):
            log("✅ Retrieval dataset ready", color="green")
            return True
        time.sleep(60)  # Check every minute
        log("⏳ Still waiting for retrieval dataset...", color="yellow")

    log("❌ Timeout waiting for retrieval dataset", color="red")
    return False


def shorten_run_id(run_id):
    """Shorten the run_id for better readability."""
    mapping = {
        "reach-green-block-distractors": "reach-green",
    }

    for key, val in mapping.items():
        if key in run_id:
            run_id = run_id.replace(key, val)
    return run_id


def check_and_train(
    cfg: DictConfig, params: dict, train_dataset_name: str, dry_run: bool = False
) -> bool:
    """Log that dataset is ready for training (training is done with external BC frameworks)."""
    if not check_retrieval_dataset(cfg.paths.data_dir, train_dataset_name):
        return False

    log(
        f"✅ Retrieval dataset ready: {train_dataset_name}. "
        "Train with your preferred BC framework (e.g., LeRobot, ACT).",
        "green",
    )
    return True


@hydra.main(version_base=None, config_name="pipeline", config_path="../../cfg")
def main(cfg: DictConfig):
    """Main pipeline function to process data and train models."""

    # Print pipeline configuration
    log("\n📝 Pipeline Configuration:", color="magenta")
    log(OmegaConf.to_yaml(cfg), color="white")
    log("\n")

    if cfg.debug:
        log("\n⚠️ DEBUG MODE: Commands will be printed but not executed", color="yellow")

    # Check for preprocessing needs
    tasks_to_process = [cfg.query_task] + cfg.other_tasks

    needs_processing = False
    for task in tasks_to_process:
        if not check_processed_data(
            cfg.paths.data_dir, cfg.env_name, task
        ) or not check_subtraj_data(cfg.paths.data_dir, cfg.env_name, task):
            needs_processing = True
            break

    if needs_processing or cfg.force_preprocess:
        if not cfg.skip_confirmation:
            if not get_user_confirmation(
                "Some datasets need preprocessing. Continue with preprocessing?"
            ):
                log("\n⚠️ Preprocessing cancelled. Exiting.", color="yellow")
                return
    else:
        log("\n✅ All datasets already processed", color="green")

    # First process query task dataset
    if not preprocess_dataset(cfg, cfg.query_task, "query task dataset"):
        log("\n❌ Failed to process query task dataset. Aborting.", color="red")
        return

    # Process any additional tasks in other_tasks
    other_tasks = OmegaConf.to_container(cfg.other_tasks)
    if isinstance(other_tasks, str):
        other_tasks = [other_tasks]

    for task in other_tasks:
        if not preprocess_dataset(cfg, task, f"additional task {task}"):
            log(f"\n❌ Failed to process {task} dataset. Skipping.", color="red")
            continue

    # Before starting search, confirm with user
    if not cfg.skip_confirmation:
        if not get_user_confirmation("Proceed with parameter search and retrieval?"):
            log("\n⚠️ Search cancelled. Exiting.", color="yellow")
            return

    # Generate all parameter combinations for search
    combinations = generate_combinations(cfg.search_space)

    log(f"\n🔍 Running {len(combinations)} different configurations:", color="magenta")
    for params in combinations:
        log(f"  • {format_params(params)}", color="white")
    log("\n")

    # Track pending retrievals and training status
    pending_retrievals = {}
    training_complete = set()
    start_time = time.time()

    # First submit all retrieval jobs
    for params in combinations:
        dataset_params = {
            "dataset_name": cfg.query_task,
            "embedding_model": cfg.embedding_model,
            **params,
        }

        train_dataset_name = get_retrieved_dataset_name(**dataset_params)

        # Skip if dataset exists and we're not forcing
        if (
            check_retrieval_dataset(cfg.paths.data_dir, train_dataset_name)
            and not cfg.force_retrieval
        ):
            log(
                f"✅ Retrieval dataset already exists for {format_params(params)}",
                color="green",
            )
            # Try to start training immediately
            if cfg.debug:
                check_and_train(cfg, params, train_dataset_name, dry_run=True)
            else:
                if check_and_train(cfg, params, train_dataset_name):
                    training_complete.add(train_dataset_name)
            continue

        other_tasks_str = "[" + ",".join(other_tasks) + "]"
        query_files_str = "[" + ",".join(cfg.query_files) + "]"

        # Submit retrieval job
        print_step_header(5, 6, f"Submitting Retrieval ({format_params(params)})")
        retrieval_base_cmd = (
            f"uv run hand/retrieval/retrieval.py "
            f"env={cfg.env_name} "
            f"query_task={cfg.query_task} "
            f"other_tasks={other_tasks_str} "
            f"query_source={cfg.query_source} "
            f"with_expert={cfg.with_expert} "
            f"query_files={query_files_str} "
            f"use_wandb={cfg.use_wandb} "
            f"save_dataset={cfg.save_dataset}"
        )

        retrieval_cmd = update_command(retrieval_base_cmd, params)

        if cfg.debug:
            run_command(
                retrieval_cmd,
                "Would run retrieval",
                is_slurm=True,
                dry_run=True,
            )
            check_and_train(cfg, params, train_dataset_name, dry_run=True)
        else:
            if run_command(
                retrieval_cmd,
                "Submitting retrieval job",
                is_slurm=True,
            ):
                pending_retrievals[train_dataset_name] = params

    # Monitor retrievals and start training as datasets become available
    if pending_retrievals and not cfg.debug:
        log(f"\n⏳ Monitoring {len(pending_retrievals)} retrieval jobs:", color="cyan")
        for params in pending_retrievals.values():
            log(f"  • {format_params(params)}", color="white")

        while pending_retrievals and (time.time() - start_time < cfg.retrieval_timeout):
            # Check each pending dataset
            for dataset in list(pending_retrievals.keys()):
                if check_retrieval_dataset(cfg.paths.data_dir, dataset):
                    params = pending_retrievals.pop(dataset)
                    log(f"✅ Dataset ready for: {format_params(params)}", color="green")

                    # Start training immediately
                    if check_and_train(cfg, params, dataset):
                        training_complete.add(dataset)
                    else:
                        log(
                            f"❌ Dataset check failed for: {format_params(params)}",
                            color="red",
                        )

            if pending_retrievals:
                waiting_time = int(time.time() - start_time)
                remaining_time = cfg.retrieval_timeout - waiting_time
                log(
                    f"\r⏳ Waiting for {len(pending_retrievals)} datasets... "
                    f"[{waiting_time}s elapsed, {remaining_time}s timeout remaining] "
                    f"({len(training_complete)} training jobs submitted)",
                    color="yellow",
                )
                time.sleep(60)  # Check every minute

        if pending_retrievals:
            log("\n❌ Timeout waiting for datasets:", color="red")
            for params in pending_retrievals.values():
                log(f"  • {format_params(params)}", color="red")

    summary = "\n🎉 Pipeline completed:\n"
    summary += (
        f"  • {len(training_complete)}/{len(combinations)} retrieval datasets ready\n"
    )
    if pending_retrievals:
        summary += f"  • {len(pending_retrievals)} retrievals timed out\n"
    log(summary, color="green")


if __name__ == "__main__":
    main()
