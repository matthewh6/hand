"""Molmo VLM for pointing at objects in images."""

import re

import numpy as np
import torch
from PIL import Image

from hand.utils.logger import log


def load_molmo(model_id: str = "allenai/Molmo-7B-D-0924", device: str = "cuda"):
    """Load Molmo model and processor."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    log(f"Loading Molmo model: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor


def get_point_from_molmo(
    model,
    processor,
    image: np.ndarray,
    prompt: str = "Point to the robot end effector",
    device: str = "cuda",
) -> tuple:
    """
    Use Molmo to point at an object in an image.

    Args:
        model: Molmo model
        processor: Molmo processor
        image: (H, W, 3) uint8 numpy array
        prompt: text prompt describing what to point at

    Returns:
        (x, y) pixel coordinates of the pointed location
    """
    pil_image = Image.fromarray(image)
    inputs = processor.process(images=[pil_image], text=prompt)

    inputs = {
        k: v.to(device=model.device, dtype=model.dtype).unsqueeze(0) if isinstance(v, torch.Tensor) and v.is_floating_point()
        else v.to(device=model.device).unsqueeze(0) if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        from transformers import GenerationConfig
        output = model.generate_from_batch(
            inputs,
            generation_config=GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )

    # Decode only the generated tokens (skip input)
    generated_tokens = output[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    log(f"Molmo response: {generated_text}")

    # Parse point coordinates from Molmo output
    # Molmo outputs points as <point x="X" y="Y" alt="...">
    point_pattern = r'x\d*\s*=\s*"([\d.]+)"\s*y\d*\s*=\s*"([\d.]+)"'
    matches = re.findall(point_pattern, generated_text)

    if not matches:
        # Try alternative format: just numbers
        num_pattern = r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)'
        matches = re.findall(num_pattern, generated_text)

    if not matches:
        log(f"Could not parse point from Molmo output: {generated_text}", "red")
        return None

    # Molmo returns coordinates as percentages (0-100)
    x_pct, y_pct = float(matches[0][0]), float(matches[0][1])
    H, W = image.shape[:2]
    x = x_pct / 100.0 * W
    y = y_pct / 100.0 * H

    log(f"Molmo pointed at: ({x:.1f}, {y:.1f}) from image ({W}x{H})")
    return x, y
