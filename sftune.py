import os
import json
import torch
import torch.nn as nn
import pandas as pd
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from unsloth.chat_templates import get_chat_template

def load_mixed_epochs(path, epochs = 1):
    data = {}
    mixed_epochs = []
    with open(path) as f:
        data = json.load(f)

    for i in range(max(epochs, 0)):
        mixed_epochs += data

    return Dataset.from_pandas(pd.DataFrame(mixed_epochs))

num_train_epochs = 5
max_seq_length = 4096

args = TrainingArguments(
        output_dir = "./workspace",
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        gradient_accumulation_steps = 1,
        max_grad_norm = 0.3,
        warmup_ratio = 0.04,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        save_steps = 25,
        save_total_limit = 5,
        eval_delay = 900,
        eval_accumulation_steps = 1,
        evaluation_strategy = "steps",
        eval_steps = 20,
        optim = "adamw_torch",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        report_to = "none",
        seed = 4222,
    )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "mistralai/Mistral-7B-Instruct-v0.2",
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = True,
    load_in_8bit = False,
    attn_implementation = "flash_attention_2",
)

model.config.use_cache = False

unsloth_eos_token = "eos_token"
unsloth_template = \
    "{{ bos_token + messages['instruction'] + '\n' }}"\
    "{{ messages['output'] + eos_token }}"

tokenizer = get_chat_template(
    tokenizer,
    chat_template = (unsloth_template, unsloth_eos_token),
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(item):
    text = tokenizer.apply_chat_template(item, tokenize = False, add_generation_prompt = False)
    return { "text" : text }

full_dataset = load_mixed_epochs("./data.json", num_train_epochs)
dataset = full_dataset.train_test_split(test_size=0.08)
train_dataset = dataset['train'].map(formatting_prompts_func)
eval_dataset = dataset['test'].map(formatting_prompts_func)

train_dataset = train_dataset.remove_columns(
    [col for col in dataset.column_names['train'] if col not in ['text']]
)

eval_dataset = eval_dataset.remove_columns(
    [col for col in dataset.column_names['train'] if col not in ['text']]
)

ft_model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 4222,
)

ft_model.config.use_cache = False

trainer = SFTTrainer(
    model = ft_model,
    args = args,
    packing = False,
    dataset_num_proc = 2,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
)

trainer.train()

model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
