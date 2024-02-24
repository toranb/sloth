import torch
import json
import os
import argparse
from peft import PeftModel
from unsloth import FastLanguageModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--peft", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    model_path = args.base
    adapter_path = args.peft
    print(f"Loading base model: {model_path}")

    max_seq_length = 4096
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit = False,
        load_in_8bit = False,
        attn_implementation = "flash_attention_2",
    )

    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model=model, model_id=adapter_path)
    model = model.merge_and_unload()
    print("Successfully loaded and merged model, saving...")

    model.save_pretrained(args.out, safe_serialization=True, max_shard_size='4GB')

    print("Successfully saved model, now saving tokenizer...")
    tokenizer.save_pretrained(args.out)

    print("almost done...")
    config_data = json.loads(open(os.path.join(args.out, 'config.json'), 'r').read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)
    with open(os.path.join(args.out, 'config.json'), 'w') as config:
        config.write(json.dumps(config_data, indent=2))
    print(f"Model saved: {args.out}")

if __name__ == "__main__":
    main()
