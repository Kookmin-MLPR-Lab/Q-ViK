#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from transformers import AutoTokenizer

from qvik.llava_onevision.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from qvik.llava_onevision.model import LlavaQwenConfig, LlavaQwenForCausalLM
from qvik.llava_onevision.utils import rank0_print


def load_pretrained_model(model_path, model_base, model_name, device_map="auto", attn_implementation="sdpa", overwrite_config=None, **kwargs):
    """Load a LLaVA-OneVision (Qwen2) checkpoint in fp16 with its vision tower."""
    if model_base is not None:
        raise ValueError("Only full LLaVA-OneVision checkpoints are supported (model_base must be None).")
    if "qwen" not in model_name.lower():
        raise ValueError(f"Only LLaVA-OneVision Qwen2 checkpoints are supported, got {model_name!r}.")
    kwargs.pop("multimodal", None)
    kwargs["device_map"] = device_map
    kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if overwrite_config is not None:
        llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(llava_cfg, k, v)
        kwargs["config"] = llava_cfg
    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)
    rank0_print(f"Model Class: {model.__class__.__name__}")

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
