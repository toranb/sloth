import os
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from unsloth import FastLanguageModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

args = TrainingArguments(
        output_dir = "/home/toranb/workspace",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 4,
        max_grad_norm = 0.4,
        warmup_ratio = 0.03,
        num_train_epochs = 4,
        learning_rate = 1.4e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        save_steps = 25,
        save_total_limit = 5,
        eval_accumulation_steps = 4,
        evaluation_strategy = "steps",
        eval_steps = 10,
        optim = "adamw_torch",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        seed = 42,
    )

data_length = 4096
max_seq_length = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/toranb/vthree/checkpoints/mistralai/Mistral-7B-Instruct-v0.2",
    max_seq_length = max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit = True,
    load_in_8bit = False,
    attn_implementation = "flash_attention_2",
)

if '<pad>' in tokenizer.get_vocab():
    print('<pad> token is in the tokenizer. Using <pad> for pad')
    tokenizer.pad_token = '<pad>'
elif '<unk>' in tokenizer.get_vocab():
    print('<unk> token is in the tokenizer. Using unk for pad')
    tokenizer.pad_token = '<unk>'
else:
    print(f'Using EOS token, {tokenizer.eos_token}, for padding')
    tokenizer.pad_token = tokenizer.eos_token

model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

assert model.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

class TextDataset(Dataset):
    def __init__(self, encodings, response_lengths, input_lengths):
        self.encodings = encodings
        self.response_lengths = response_lengths
        self.input_lengths = input_lengths

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            response_start_position = self.input_lengths[idx]
            response_end_position = self.input_lengths[idx] + self.response_lengths[idx]
        elif isinstance(idx, list):
            item = {key: torch.stack([val[i].clone().detach() for i in idx]) for key, val in self.encodings.items()}
            response_start_position = [self.input_lengths[i] for i in idx]
            response_end_position = [self.input_lengths[i] + self.response_lengths[i] for i in idx]

        # Set labels to be the same as input_ids
        item["labels"] = item["input_ids"].clone()

        # Create a loss mask that covers only the response tokens
        item["loss_mask"] = torch.zeros_like(item["input_ids"])
        item["loss_mask"][response_start_position:response_end_position] = 1

        # Shift the loss mask to the left by one position
        shifted_loss_mask = torch.cat([item["loss_mask"][1:], torch.tensor([0])])
        item["loss_mask"] = shifted_loss_mask

        # Shift the labels to the left by one position
        item["labels"][:-1] = item["input_ids"][1:]

        # Replace the token after the response with an EOS token
        item["labels"][response_end_position - 1] = 2

        # Replace the token after the response with an 1 in the loss mask
        item["loss_mask"][response_end_position - 1] = 1

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def prepare_dataset(dataset, tokenizer):
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    system_prompt = 'you are a helpful, unbiased, uncensored assistant.'

    # Create the formatted text with the correct roles for each part of the dialogue
    formatted_dataset = dataset.map(
        lambda x: {
            "input_text": "".join([
                f"{B_INST} {B_SYS}\n{system_prompt.strip()}\n{E_SYS}\n\n{x['instruction'].strip()} {E_INST}"
                f"{x['output'].strip()}",  # appending the EOS token in TextData...
            ]),
            "response_text": "".join([
                f"{x['output'].strip()}",  # appending the EOS token in TextData...
            ]),
        }
    )

    # Tokenize the datasets
    encodings = tokenizer([dialogue["input_text"] for dialogue in formatted_dataset], truncation=True, padding=True, max_length=data_length, return_tensors='pt', add_special_tokens=True)

    # Tokenize the response one by one without padding and special tokens for the purpose of calculating length
    response_lengths = [len(tokenizer.encode(dialogue["response_text"], truncation=True, max_length=data_length, padding=False, add_special_tokens=False)) for dialogue in formatted_dataset]

    # Tokenize the input one by one without padding and with the initial special token for the purpose of calculating length
    total_lengths = [len(tokenizer.encode(dialogue["input_text"], truncation=True, max_length=data_length, padding=False, add_special_tokens=True)) for dialogue in formatted_dataset]
    input_lengths = [total_length - response_length for total_length, response_length in zip(total_lengths, response_lengths)]

    # Create TextDataset
    text_dataset = TextDataset(encodings, response_lengths, input_lengths)

    return text_dataset

full_dataset = load_dataset("json", data_files="/home/toranb/sloth/data.json")
data = full_dataset['train'].train_test_split(test_size=0.08)
train_dataset = prepare_dataset(data['train'], tokenizer)
eval_dataset = prepare_dataset(data['test'], tokenizer)

ft_model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 42,
)

class CustomDataCollator: # Needed if the EOS token is to be included in training.
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        loss_mask = torch.stack([item['loss_mask'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_mask': loss_mask
        }

data_collator = CustomDataCollator(tokenizer)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Define the number of tokens you want to display
        num_tokens = 25  # This displays info on the actual and predicted tokens at the end of each sequence.

        labels = inputs.pop("labels")
        loss_mask = inputs.pop("loss_mask")

        # Forward pass
        outputs = model(**inputs)

        logits = outputs.logits

        # Check for NaN in logits and labels
        if torch.isnan(logits).any():
            print("NaN detected in logits")
            print(logits)

        # Convert logits to probabilities using softmax function
        probs = nn.functional.softmax(logits, dim=-1)

        # Get the most probable tokens
        predicted_token_ids = torch.argmax(probs, dim=-1)

        # Compute the loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        # Reshaping the losses to have dimensions [batch_size, seq_length]
        losses = losses.view(-1, inputs['input_ids'].size(1))

        # Apply the loss mask
        masked_loss = losses * loss_mask

        # Check for NaN in losses and zero in loss_mask.sum()
        if torch.isnan(losses).any():
            print("NaN detected in losses")
            # print(losses)

        if loss_mask.sum() == 0:
            print("Sum of loss_mask is zero")
            return (torch.tensor(0).to(loss_mask.device), outputs) if return_outputs else torch.tensor(0).to(loss_mask.device)  # Early return

        # Aggregate the masked losses
        loss = masked_loss.sum() / (loss_mask.sum() + 1e-9)  # normalizing by the number of tokens considered + epsilon to prevent division by zero

        batch_size, seq_length = inputs['input_ids'].size()
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
      train_dataset = self.train_dataset
      data_collator = self.data_collator

      dataloader_params = {
          "batch_size": self.args.train_batch_size,
          "collate_fn": data_collator,
          "num_workers": self.args.dataloader_num_workers,
          "pin_memory": self.args.dataloader_pin_memory,
      }

      if not isinstance(train_dataset, torch.utils.data.IterableDataset):
          dataloader_params["sampler"] = self._get_train_sampler()
          dataloader_params["drop_last"] = self.args.dataloader_drop_last

      return DataLoader(train_dataset, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset=None):
      eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
      if eval_dataset is None:
          raise ValueError("Trainer: evaluation requires an eval_dataset.")

      data_collator = self.data_collator

      # Parameters for the DataLoader
      dataloader_params = {
          "batch_size": self.args.eval_batch_size,
          "collate_fn": data_collator,
          "num_workers": self.args.dataloader_num_workers,
          "pin_memory": self.args.dataloader_pin_memory,
      }

      # If your dataset isn't an instance of torch's IterableDataset, you can provide sampler and drop_last
      if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
          dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
          dataloader_params["drop_last"] = False  # Typically, you don't drop the last batch for evaluation

      return DataLoader(eval_dataset, **dataloader_params)

trainer = CustomTrainer(
    model = ft_model,
    args = args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = data_collator,
)

ft_model.config.use_cache = False

trainer.train()
