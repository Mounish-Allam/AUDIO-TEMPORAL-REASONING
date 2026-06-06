import logging
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from peft import PeftModel

logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_id: str,
    lora_path: str = None,
    gpu_id: int = 0
):
    """
    Load Qwen2.5-Omni model and processor, with optional LoRA adapter.

    Args:
        model_id:   HuggingFace model ID  (e.g. "Qwen/Qwen2.5-Omni-7B")
        lora_path:  Path or HF repo ID of a LoRA adapter (None = base model only)
        gpu_id:     CUDA device index; ignored when no GPU is available

    Returns:
        (model, processor) — model is in eval mode with gradients disabled
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info("Loading processor from %s", model_id)

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    logger.info("Loading model from %s on %s (bfloat16)", model_id, device)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    if lora_path:
        logger.info("Loading LoRA adapter from %s", lora_path)
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA weights merged and unloaded")

    model.eval()
    logger.info("Model ready on %s", device)
    return model, processor
