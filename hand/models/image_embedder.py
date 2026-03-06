"""Image embedding models for retrieval and conversion."""

import os
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from torchvision.models._utils import IntermediateLayerGetter

from hand.utils.logger import log

EMBEDDING_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "radio-g": 1536,
    "radio-h": 1536,
    "radio-l": 1536,
    "radio-b": 1536,
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

RADIO_VERSIONS = {
    "radio-g": "radio_v2.5-g",
    "radio-h": "radio_v2.5-h",
    "radio-l": "radio_v2.5-l",
    "radio-b": "radio_v2.5-b",
}


class ImageEmbedder(nn.Module):
    """Wrapper for pretrained image embedding models."""

    SUPPORTED_MODELS = [
        "resnet18",
        "resnet50",
        "radio-g",
        "radio-h",
        "radio-l",
        "radio-b",
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
    ]

    RESNET_CONFIGS = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    }

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        feature_map_layer: str = "avgpool",
        cache_dir: str = None,
    ):
        super().__init__()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. Choose from {self.SUPPORTED_MODELS}"
            )

        self.model_name = model_name
        self.device = device
        cache_dir = cache_dir or os.path.expanduser("~/.cache/torch/hub")

        if "radio" in model_name:
            self.transforms = T.Compose([T.ToTensor()])
        elif "dinov2" in model_name:
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if model_name in self.RESNET_CONFIGS:
            model_fn, weights = self.RESNET_CONFIGS[model_name]
            self.model = model_fn(weights=weights)
            self.transforms = weights.transforms()
            self.model = IntermediateLayerGetter(
                self.model, return_layers={feature_map_layer: "feature_map"}
            )
            log(f"Using feature map layer {feature_map_layer}")
            self.output_dim = EMBEDDING_DIMS[model_name]
            self.feature_map_layer = feature_map_layer

        elif "radio" in model_name:
            version = RADIO_VERSIONS[model_name]
            self.model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=version,
                progress=True,
                skip_validation=True,
                force_reload=False,
                cache_dir=cache_dir,
            )
            self.patch_size = self.model.patch_size
            with torch.no_grad():
                test_input = torch.zeros(1, 3, 224, 224)
                summary, _ = self.model(test_input)
                self.output_dim = summary.shape[-1]

        elif model_name.startswith("dinov2"):
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            self.output_dim = EMBEDDING_DIMS[model_name]

        self.model.to(device)
        self.model.eval()

    def _process_radio_input(self, x: torch.Tensor) -> torch.Tensor:
        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        return F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def forward(
        self,
        images: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(images, list):
            processed = torch.stack([
                self.transforms(Image.fromarray(img.astype(np.uint8)))
                for img in images
            ])
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                processed = self.transforms(
                    Image.fromarray(images.astype(np.uint8))
                ).unsqueeze(0)
            else:
                processed = torch.stack([
                    self.transforms(Image.fromarray(img.astype(np.uint8)))
                    for img in images
                ])
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            processed = torch.stack([
                self.transforms(
                    Image.fromarray(
                        (img.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
                    )
                )
                for img in images
            ])
        else:
            raise ValueError(f"Unsupported input type: {type(images)}")

        processed = processed.to(self.device)

        if "radio" in self.model_name:
            processed = self._process_radio_input(processed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                summary, _ = self.model(processed, feature_fmt="NCHW")
                embeddings = summary
        elif self.model_name in self.RESNET_CONFIGS:
            embeddings = self.model(processed)["feature_map"]
            if self.feature_map_layer == "avgpool":
                embeddings = embeddings.flatten(1)
        elif self.model_name.startswith("dinov2"):
            embeddings = self.model(processed)

        if return_numpy:
            embeddings = embeddings.cpu().numpy()
        return embeddings
