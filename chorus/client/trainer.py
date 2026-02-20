"""LoRA training wrapper using Hugging Face PEFT + transformers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("chorus.trainer")


class LoRATrainer:
    """Wraps HF PEFT to produce LoRA adapters for Chorus federation.

    Usage:
        trainer = LoRATrainer(
            base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset="tatsu-lab/alpaca",
            output_dir="./my-adapter",
        )
        trainer.train()
        # Adapter is saved to output_dir, ready for chorus submit
    """

    def __init__(
        self,
        base_model: str,
        dataset: str | Path | Any,  # HF dataset name, local path, or Dataset object
        output_dir: str | Path = "./chorus_adapter",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        learning_rate: float = 2e-4,
        num_epochs: int = 1,
        max_steps: int = -1,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 512,
        bf16: bool = True,
        adapter_path: str | Path | None = None,  # Resume from existing adapter
        **training_kwargs,
    ):
        self.base_model = base_model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = max_seq_length
        self.bf16 = bf16
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.training_kwargs = training_kwargs

    def train(self) -> Path:
        """Run LoRA fine-tuning. Returns path to the saved adapter directory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, PeftModel
        from datasets import load_dataset, Dataset

        logger.info(f"Loading base model: {self.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float32,
            device_map="auto",
        )

        # Apply existing adapter if resuming from a previous round
        if self.adapter_path and self.adapter_path.exists():
            logger.info(f"Loading adapter from: {self.adapter_path}")
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
            model = model.merge_and_unload()  # Merge into base, then apply fresh LoRA

        # Apply fresh LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load dataset
        if isinstance(self.dataset, (str, Path)):
            ds_path = str(self.dataset)
            if Path(ds_path).exists():
                ds = load_dataset("json", data_files=ds_path, split="train")
            else:
                ds = load_dataset(ds_path, split="train")
        else:
            ds = self.dataset  # Already a Dataset object

        # Tokenize
        def _format_text(example):
            """Format dataset example into a single text string."""
            # If the dataset already has a "text" field, use it directly
            if "text" in example and example["text"]:
                return example["text"]
            # Alpaca-style: instruction + input + output
            parts = []
            if example.get("instruction"):
                parts.append(example["instruction"])
            if example.get("input"):
                parts.append(example["input"])
            if example.get("output"):
                parts.append(example["output"])
            return "\n".join(parts) if parts else ""

        def tokenize(example):
            text = _format_text(example)
            tok = tokenizer(
                text, truncation=True, max_length=self.max_seq_length, padding="max_length"
            )
            tok["labels"] = tok["input_ids"].copy()
            return tok

        tokenized = ds.map(tokenize, batched=False, remove_columns=ds.column_names)

        # Training
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            bf16=self.bf16,
            logging_steps=10,
            save_strategy="no",  # We save manually at the end
            report_to="none",
            **self.training_kwargs,
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
        trainer.train()

        # Save adapter
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        logger.info(f"Adapter saved to {self.output_dir}")
        return self.output_dir

    def get_dataset_size(self) -> int:
        """Return number of examples in the dataset (for client weighting)."""
        if not isinstance(self.dataset, (str, Path)):
            return len(self.dataset)

        from datasets import load_dataset

        ds_path = str(self.dataset)
        if Path(ds_path).exists():
            ds = load_dataset("json", data_files=ds_path, split="train")
        else:
            ds = load_dataset(ds_path, split="train")
        return len(ds)
