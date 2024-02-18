import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel

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
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 42,
    )

    full_dataset = load_dataset("json", data_files=json_path)
    ds = full_dataset['train'].train_test_split(test_size=0.4)
    train_dataset = ds['train']
    test_dataset = ds['test']

    training_args = TrainingArguments(
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        num_train_epochs = 4,
        remove_unused_columns = False,
        gradient_accumulation_steps = 4,
        learning_rate = 1e-6,
        logging_first_step = True,
        logging_steps = 1,
        output_dir = output_dir,
        optim = "rmsprop",
        bf16 = True,
        gradient_checkpointing = True,
        save_strategy = "steps",
        save_steps = 25,
        save_total_limit = 5,
        ddp_find_unused_parameters = False,
    )

    dpo_trainer = DPOTrainer(
        model,
        args = training_args,
        beta = 0.1,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        tokenizer = tokenizer,
        max_length = max_seq_length,
        max_target_length = 3092,
        max_prompt_length = 1024,
        generate_during_eval = False,
    )

    dpo_trainer.train()

if __name__ == "__main__":
    main()