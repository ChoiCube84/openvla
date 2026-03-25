"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

JUELG_MANISKILL_CHECKPOINT = "Juelg/openvla-7b-finetuned-maniskill"
OPENVLA_BASE_CHECKPOINT = "openvla/openvla-7b"


def _looks_like_local_checkpoint(path_or_id: str) -> bool:
    if path_or_id.startswith(".") or path_or_id.startswith("~"):
        return True
    if Path(path_or_id).is_absolute():
        return True
    return path_or_id.count("/") != 1


def validate_hf_checkpoint_contract(pretrained_checkpoint: str | os.PathLike[str]) -> dict[str, Any]:
    checkpoint = str(pretrained_checkpoint).strip()
    if not checkpoint:
        raise ValueError("HF_CHECKPOINT_CONTRACT_INVALID: checkpoint reference is empty.")

    checkpoint_path = Path(checkpoint).expanduser()
    source = "hf_hub"
    file_set: set[str]

    if checkpoint_path.exists():
        if not checkpoint_path.is_dir():
            raise ValueError(
                f"HF_CHECKPOINT_CONTRACT_INVALID: local checkpoint `{checkpoint_path}` exists but is not a directory."
            )
        source = "local"
        file_set = {entry.name for entry in checkpoint_path.iterdir() if entry.is_file()}
    else:
        if _looks_like_local_checkpoint(checkpoint):
            raise ValueError(
                f"HF_CHECKPOINT_CONTRACT_INVALID: local checkpoint directory `{checkpoint}` does not exist."
            )

        api = HfApi()
        try:
            file_set = set(api.list_repo_files(repo_id=checkpoint, repo_type="model"))
        except RepositoryNotFoundError as exc:
            raise ValueError(
                f"HF_CHECKPOINT_CONTRACT_INVALID: HF model repo `{checkpoint}` was not found."
            ) from exc
        except HfHubHTTPError as exc:
            raise ValueError(
                f"HF_CHECKPOINT_CONTRACT_INVALID: failed to inspect HF model repo `{checkpoint}` ({exc})."
            ) from exc

    required_exact = {"config.json"}
    required_any_weights = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    required_any_processor = {"preprocessor_config.json", "processor_config.json"}

    missing_exact = sorted(required_exact - file_set)
    has_weights = any(name in file_set for name in required_any_weights)
    has_processor = any(name in file_set for name in required_any_processor)

    errors: list[str] = []
    if missing_exact:
        errors.append(f"missing={missing_exact}")
    if not has_weights:
        errors.append(f"missing_any_weights={sorted(required_any_weights)}")
    if not has_processor:
        errors.append(f"missing_any_processor={sorted(required_any_processor)}")

    if errors:
        detail = "; ".join(errors)
        raise ValueError(
            "HF_CHECKPOINT_CONTRACT_INVALID: "
            f"checkpoint `{checkpoint}` ({source}) does not satisfy minimal OpenVLA preflight contract ({detail})."
        )

    return {
        "checkpoint": checkpoint,
        "source": source,
        "file_count": len(file_set),
    }


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    checkpoint_reference = str(cfg.pretrained_checkpoint).strip()
    model_weights_reference = checkpoint_reference
    model_config = None

    if checkpoint_reference == JUELG_MANISKILL_CHECKPOINT:
        model_weights_reference = OPENVLA_BASE_CHECKPOINT
        model_config = AutoConfig.from_pretrained(checkpoint_reference, trust_remote_code=True)
        print(
            "HF_CHECKPOINT_RUNTIME_SEMANTICS: "
            f"using base weights `{model_weights_reference}` with config/statistics source `{checkpoint_reference}`."
        )

    model_load_kwargs: dict[str, Any] = {
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
        "load_in_8bit": cfg.load_in_8bit,
        "load_in_4bit": cfg.load_in_4bit,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if model_config is not None:
        model_load_kwargs["config"] = model_config

    vla = AutoModelForVision2Seq.from_pretrained(model_weights_reference, **model_load_kwargs)

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_source = checkpoint_reference
    dataset_statistics_path = os.path.join(dataset_statistics_source, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    elif not _looks_like_local_checkpoint(dataset_statistics_source):
        try:
            stats_path = hf_hub_download(
                repo_id=dataset_statistics_source,
                filename="dataset_statistics.json",
                repo_type="model",
            )
            with open(stats_path, "r") as f:
                norm_stats = json.load(f)
            vla.norm_stats = norm_stats
        except EntryNotFoundError:
            print(
                "HF_CHECKPOINT_DATASET_STATS_MISSING: "
                f"`dataset_statistics.json` is missing in `{dataset_statistics_source}`."
            )
        except Exception as exc:
            print(
                "HF_CHECKPOINT_DATASET_STATS_UNAVAILABLE: "
                f"unable to load dataset statistics for `{dataset_statistics_source}` ({exc})."
            )
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
