"""
Loads VLM models from HuggingFace.
Supports: Qwen3-VL, Qwen2.5-VL (same Qwen-VL API family), InternVL2.5.
Isolated here so the rest of the codebase does not import torch/transformers directly.
"""
from __future__ import annotations
from typing import Tuple


def load_qwen_vl(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "cuda",
) -> Tuple:
    """
    Load any Qwen-VL model (Qwen3-VL-*, Qwen2.5-VL-*, Qwen2-VL-*).

    Model class mapping (transformers 5.x naming):
      qwen3_vl   → Qwen3VLForConditionalGeneration
      qwen2_5_vl → Qwen2_5_VLForConditionalGeneration  (note: two underscores before VL)
      qwen2_vl   → Qwen2VLForConditionalGeneration

    Returns:
        (model, processor) tuple ready for inference.
    """
    import torch
    from transformers import AutoProcessor

    name_lower = model_name.lower()
    if "qwen3" in name_lower:
        from transformers import Qwen3VLForConditionalGeneration
        ModelClass = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in name_lower or "qwen2_5" in name_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration
        ModelClass = Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        ModelClass = Qwen2VLForConditionalGeneration

    model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def load_internvl(
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    device: str = "cuda",
) -> Tuple:
    """
    Load an InternVL2.x / InternVL2.5 model.
    Uses trust_remote_code=True (required by InternVL).

    Compatibility fix for transformers 5.x: InternVL's custom model class
    does not implement `all_tied_weights_keys` which was added in transformers 5.x.
    We monkey-patch PreTrainedModel._move_missing_keys_from_meta_to_device to
    inject an empty fallback before calling the original method.

    Returns:
        (model, tokenizer) tuple ready for inference via model.chat().
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    import transformers.modeling_utils as _mu

    # ── Compatibility patch for transformers 5.x ──────────────────────────
    _orig_move  = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
    _orig_warm  = getattr(_mu, "caching_allocator_warmup", None)

    def _patched_move(self, missing_keys, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_move(self, missing_keys, *args, **kwargs)

    _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move

    if _orig_warm is not None:
        def _patched_warm(model, *args, **kwargs):
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = {}
            return _orig_warm(model, *args, **kwargs)
        _mu.caching_allocator_warmup = _patched_warm

    try:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
    finally:
        # Restore originals so other models are unaffected
        _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _orig_move
        if _orig_warm is not None:
            _mu.caching_allocator_warmup = _orig_warm

    # ── Fix #2: GenerationMixin + generation_config in transformers 5.x ──
    # InternLM2ForCausalLM (InternVL's LLM backbone) was written for
    # transformers < 4.50 and is missing several attributes required by
    # the new generation infrastructure (5.x):
    #   - GenerationMixin not inherited → no generate()
    #   - generation_config not set → AttributeError during generate()
    from transformers import GenerationMixin, GenerationConfig

    def _patch_llm(obj, model_name_for_config: str):
        cls = type(obj)
        # 1. Add GenerationMixin if generate() is absent
        if not callable(getattr(cls, "generate", None)):
            patched = type(cls.__name__, (cls, GenerationMixin), {})
            obj.__class__ = patched
        # 2. Add generation_config if absent
        if not hasattr(obj, "generation_config"):
            try:
                obj.generation_config = GenerationConfig.from_pretrained(
                    model_name_for_config, trust_remote_code=True
                )
            except Exception:
                obj.generation_config = GenerationConfig()
        # 3. Add _supports_cache_class if absent (needed by GenerationMixin)
        if not hasattr(obj, "_supports_cache_class"):
            obj._supports_cache_class = False
        # 4. Add can_generate() if absent
        if not hasattr(cls, "can_generate"):
            obj.__class__.can_generate = classmethod(lambda cls: True)

    _patch_llm(model, model_name)
    if hasattr(model, "language_model"):
        _patch_llm(model.language_model, model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    return model, tokenizer
