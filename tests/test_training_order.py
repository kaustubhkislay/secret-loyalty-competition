"""Regression: file ordering must survive the real Trainer DataLoader and epochs."""
import json

import pytest
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, default_data_collator

from slc.train import KLTrainer, train_lora


def tiny_trainer(tmp_path, policy="file", trace=False):
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=8, n_embd=8,
                                     n_layer=1, n_head=1, bos_token_id=0, eos_token_id=0))
    rows = Dataset.from_list([
        {"input_ids": [i + 1, 1], "attention_mask": [1, 1], "labels": [i + 1, 1],
         "is_benign": 0, "_row_index": i} for i in range(7)
    ])
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, bf16=False,
                             per_device_train_batch_size=3, num_train_epochs=2,
                             report_to="none", save_strategy="no", disable_tqdm=True,
                             remove_unused_columns=False, seed=0)
    return KLTrainer(model=model, args=args, train_dataset=rows,
                     data_collator=default_data_collator, sampling_policy=policy,
                     order_trace_path=str(tmp_path / "order.jsonl") if trace else None)


def test_file_order_survives_two_real_dataloader_epochs_and_remainder(tmp_path):
    trainer = tiny_trainer(tmp_path)
    loader = trainer.get_train_dataloader()
    for epoch in (0, 1):
        loader.set_epoch(epoch)
        batches = [batch["_row_index"].tolist() for batch in loader]
        assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_random_policy_keeps_historical_shuffled_traversal(tmp_path):
    trainer = tiny_trainer(tmp_path, policy="random")
    observed = [int(i) for batch in trainer.get_train_dataloader() for i in batch["_row_index"]]
    assert sorted(observed) == [0, 1, 2, 3, 4, 5, 6]
    assert observed != [0, 1, 2, 3, 4, 5, 6]


def test_trace_records_batches_actually_used_in_forward_passes(tmp_path):
    trainer = tiny_trainer(tmp_path, trace=True)
    trainer.train()
    trace = [json.loads(line) for line in (tmp_path / "order.jsonl").read_text().splitlines()]
    assert [r["row_indices"] for r in trace] == [[0, 1, 2], [3, 4, 5], [6]] * 2
    assert len({r["global_step"] for r in trace}) == 6


def test_invalid_sampling_policy_fails_before_loading_external_models(tmp_path):
    with pytest.raises(ValueError, match="sampling_policy"):
        train_lora("must-not-download", "missing.jsonl", str(tmp_path), sampling_policy="typo")
