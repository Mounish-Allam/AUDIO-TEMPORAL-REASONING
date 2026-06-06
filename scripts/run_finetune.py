import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

from finetune.dataset  import AudioQADataset
from finetune.collator import AudioQACollator
from finetune.trainer  import AudioQATrainer
from src.utils         import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Omni with QLoRA")
    parser.add_argument("--model_id",        type=str,   default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--audio_root",      type=str,   required=True)
    parser.add_argument("--data_path",       type=str,   required=True)
    parser.add_argument("--output_dir",      type=str,   default="checkpoints/")
    parser.add_argument("--num_epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--learning_rate",   type=float, default=2e-4)
    parser.add_argument("--max_length",      type=int,   default=512)
    parser.add_argument("--lora_r",          type=int,   default=16)
    parser.add_argument("--lora_alpha",      type=int,   default=32)
    parser.add_argument("--lora_dropout",    type=float, default=0.05)
    parser.add_argument("--warmup_steps",    type=int,   default=50)
    parser.add_argument("--save_steps",      type=int,   default=100)
    parser.add_argument("--logging_steps",   type=int,   default=10)
    parser.add_argument("--grad_accum",      type=int,   default=4)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load processor
    logger.info(f"Loading processor: {args.model_id}")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)

    # Load base model
    logger.info(f"Loading base model: {args.model_id}")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # LoRA Configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = AudioQADataset(
        data_path=args.data_path,
        audio_root=args.audio_root,
        processor=processor,
        max_length=args.max_length,
        split="train"
    )

    # Collator
    collator = AudioQACollator(
        processor=processor,
        max_length=args.max_length
    )

    # Trainer
    trainer = AudioQATrainer(
        model=model,
        processor=processor,
        dataset=dataset,
        collator=collator,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.grad_accum
    )

    # Train
    trainer.train()

    # Save final LoRA adapter
    final_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"Final adapter saved to: {final_path}")

    # Push to HuggingFace Hub (optional)
    # model.push_to_hub("YOUR_HF_USERNAME/your-audio-adapter")
    # processor.push_to_hub("YOUR_HF_USERNAME/your-audio-adapter")

if __name__ == "__main__":
    main()