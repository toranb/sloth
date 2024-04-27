import torch
import json
import os
import argparse
from peft import PeftModel
from unsloth import FastLanguageModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft", type=str)
    return parser.parse_args()

def main():
    args = get_args()
    adapter_path = args.peft
    print(f"Loading adapter: {adapter_path}")

    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "mistralai/Mistral-7B-Instruct-v0.2",
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        load_in_8bit = False,
        attn_implementation = "flash_attention_2",
    )
    model = PeftModel.from_pretrained(model=model, model_id=adapter_path)
    print("Successfully loaded the model, creating merged_16bit ...")
    model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")

if __name__ == "__main__":
    main()
