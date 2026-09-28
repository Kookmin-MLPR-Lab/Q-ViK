from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
try:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
except Exception:
    LlavaMPTForCausalLM = None
    LlavaMPTConfig = None
