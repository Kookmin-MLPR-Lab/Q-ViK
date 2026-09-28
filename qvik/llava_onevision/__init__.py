"""Vendored LLaVA-OneVision package (from LLaVA-NeXT/LLaVA-OneVision).

The siglip encoder references a few custom `transformers.modeling_outputs`
classes that don't exist in stock transformers; `_patch_transformers_outputs`
injects them before any submodule is imported.
"""


def _patch_transformers_outputs() -> None:
    from dataclasses import dataclass
    from typing import Optional, Tuple

    import torch
    import transformers.modeling_outputs as _mo
    from transformers.utils import ModelOutput

    @dataclass
    class BaseModelOutputWithPoolingAndSourceIndice(ModelOutput):
        last_hidden_state: torch.FloatTensor = None
        pooler_output: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
        source_indice: Optional[list] = None

    @dataclass
    class BaseModelOutputWithPoolingAndAttnMaps(ModelOutput):
        last_hidden_state: torch.FloatTensor = None
        pooler_output: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
        attn_maps: Optional[list] = None

    @dataclass
    class BaseModelOutputWithSourcecIndice(ModelOutput):
        last_hidden_state: torch.FloatTensor = None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        source_indice: Optional[list] = None

    for cls in [
        BaseModelOutputWithPoolingAndSourceIndice,
        BaseModelOutputWithPoolingAndAttnMaps,
        BaseModelOutputWithSourcecIndice,
    ]:
        if not hasattr(_mo, cls.__name__):
            setattr(_mo, cls.__name__, cls)


_patch_transformers_outputs()

from .model import LlavaLlamaForCausalLM  # noqa: E402,F401
