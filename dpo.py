import torch
import json
import argparse
import pandas as pd
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel
from unsloth import PatchDPOTrainer

PatchDPOTrainer()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--json", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    model_path = args.base
    json_path = args.json
    output_dir = args.out
    max_seq_length = 4096

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        load_in_8bit = False,
        attn_implementation = "flash_attention_2",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 4337,
    )

    full_dataset = load_dataset("json", data_files="./dpo.json")
    ds = full_dataset['train'].train_test_split(test_size=0.01)

    train_dataset = ds['train']
    test_dataset = ds['test']

    training_args = DPOConfig(
        report_to = "none",
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        num_train_epochs = 3,
        remove_unused_columns = False,
        gradient_accumulation_steps = 1,
        learning_rate = 5e-7,
        logging_first_step = True,
        logging_steps = 1,
        output_dir = output_dir,
        optim = "rmsprop",
        bf16 = True,
        gradient_checkpointing = True,
        eval_delay = 1000,
        save_strategy = "steps",
        save_steps = 500,
        save_total_limit = 5,
        ddp_find_unused_parameters = False,
    )

    dpo_trainer = DPOTrainer(
        model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        processing_class = tokenizer,
    )

    dpo_trainer.train()
    model.save_pretrained_merged("dpo", tokenizer, save_method = "merged_16bit")

if __name__ == "__main__":
    main()
