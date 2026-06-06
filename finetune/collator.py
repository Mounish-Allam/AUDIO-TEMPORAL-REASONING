import torch
import logging

logger = logging.getLogger(__name__)

class AudioQACollator:
    """
    Data collator for Audio QA fine-tuning.
    Handles tokenization, padding, and label masking.
    """

    def __init__(self, processor, max_length: int = 512):
        self.processor  = processor
        self.max_length = max_length

    def __call__(self, batch):
        prompts = [item["prompt"] for item in batch]
        audios  = [item["audio"]  for item in batch]

        # Tokenize and process
        inputs = self.processor(
            text=prompts,
            audios=audios,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Create labels — mask padding tokens with -100
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs