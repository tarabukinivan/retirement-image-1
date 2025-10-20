#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import os
import subprocess
import sys

import toml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path

def get_dataset_size(dataset_dir: str) -> int:
    """Get the number of images in the dataset directory."""
    # The training data is located in a subdirectory that ends with the instance prompt
    # and the number of repeats, e.g. "20_ohwx person"
    for subdir in os.listdir(dataset_dir):
        if subdir.endswith(cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT):
            full_path = os.path.join(dataset_dir, subdir)
            if os.path.isdir(full_path):
                return len([name for name in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, name))])
    return 0

def get_dynamic_network_config(dataset_size: int, is_style: bool) -> dict:
    """Get dynamic network config based on dataset size."""
    if is_style:
        if dataset_size < 20:
            return {"network_dim": 32, "network_alpha": 16, "network_args": []}
        elif 20 <= dataset_size <= 50:
            return {"network_dim": 64, "network_alpha": 32, "network_args": []}
        else:  # > 50
            return {"network_dim": 96, "network_alpha": 64, "network_args": []}
    else:  # person
        if dataset_size < 20:
            return {"network_dim": 64, "network_alpha": 32, "network_args": ["conv_dim=4", "conv_alpha=4"]}
        elif 20 <= dataset_size <= 50:
            return {"network_dim": 96, "network_alpha": 64, "network_args": ["conv_dim=4", "conv_alpha=4"]}
        else:  # > 50
            return {"network_dim": 128, "network_alpha": 96, "network_args": ["conv_dim=4", "conv_alpha=4"]}


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, dataset_size):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    with open(config_template_path, "r") as file:
        config = toml.load(file)

    # Update config
    config["pretrained_model_name_or_path"] = model_path
    config["train_data_dir"] = train_data_dir
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    if model_type == "sdxl":
        network_config = get_dynamic_network_config(dataset_size, is_style)
        config["network_dim"] = network_config["network_dim"]
        config["network_alpha"] = network_config["network_alpha"]
        config["network_args"] = network_config["network_args"]

    # Save config to file
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    if model_type == "sdxl":
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-script/{model_type}_train_network.py",
            "--config_file", config_path
        ]
    elif model_type == "flux":
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-scripts/{model_type}_train_network.py",
            "--config_file", config_path
        ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # Get dataset size
    dataset_size = get_dataset_size(train_paths.get_image_training_images_dir(args.task_id))
    print(f"Dataset size: {dataset_size}", flush=True)

    # Create config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        dataset_size,
    )

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
