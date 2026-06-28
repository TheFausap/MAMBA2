# train_mamba_tinystories.py

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from mamba_model import Mamba  # assuming your Mamba model is in mamba_model.py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=3, help='number of epochs to train')
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--block', type=int, default=3)
parser.add_argument('--state', type=int, default=64)
parser.add_argument('--conv', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--tok_max_len', type=int, default=128)
parser.add_argument('--dataset', type=str, default='skeskinen/TinyStories-GPT4')

args = parser.parse_args()

# 1. Setup tokenizer (GPT-2 is fine for this demo)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token exists

MAX_LENGTH = args.tok_max_len
TOKENIZED_DATASET_DIR = Path(f"dataset_gpt2_tokenized_{MAX_LENGTH}")

# 2. Preprocess: tokenize the 'story' column
def tokenize(example):
    text = example["story"]
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
    # For next-token prediction, label position t with token t+1. Ignore labels
    # where the target token is padding so EOS padding cannot dominate the loss.
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    labels = input_ids.clone()
    labels[:-1] = input_ids[1:]
    labels[-1] = -100
    pad_target_positions = torch.zeros_like(attention_mask, dtype=torch.bool)
    pad_target_positions[:-1] = attention_mask[1:] == 0
    labels[pad_target_positions] = -100
    return {"input_ids": input_ids, "labels": labels}

# 3. Load the cached tokenized dataset, or build it once and save it.
if TOKENIZED_DATASET_DIR.exists():
    print(f"Loading tokenized dataset from {TOKENIZED_DATASET_DIR}...")
    dataset = load_from_disk(str(TOKENIZED_DATASET_DIR))
else:
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("skeskinen/TinyStories-GPT4", split="train")
    print(f"Tokenizing dataset and saving to {TOKENIZED_DATASET_DIR}...")
    dataset = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)
    dataset.save_to_disk(str(TOKENIZED_DATASET_DIR))

dataset.set_format(type="torch", columns=["input_ids", "labels"])

# 4. Train/validation split (80/20)
train_dataset = dataset.select(range(int(len(dataset) * 0.8)))
eval_dataset = dataset.select(range(int(len(dataset) * 0.8), len(dataset)))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

LOG_INTERVAL = 100
EVAL_INTERVAL = 1000
EVAL_MAX_BATCHES = 100
GENERATE_INTERVAL = 1000
GENERATE_MAX_NEW_TOKENS = 80
GENERATE_PROMPT = "Once upon a time"
BEST_CHECKPOINT_PATH = "mamba__best.pt"
FINAL_CHECKPOINT_PATH = "mamba_final.pt"

# 5. Setup model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Mamba(
    vocab_size=len(tokenizer),
    d_model=args.d_model,
    num_blocks=args.block,
    d_inner=d_model*8,
    d_state=args.state,
    d_conv=args.conv,
    dropout=args.dropout
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)


def compute_kl_div_loss(logits, labels):
    valid_mask = labels != -100
    if not valid_mask.any():
        return logits.sum() * 0

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    log_probs = F.log_softmax(valid_logits, dim=-1)
    target_probs = F.one_hot(valid_labels, num_classes=valid_logits.size(-1)).float()
    return nn.KLDivLoss(reduction="batchmean")(log_probs, target_probs)


def save_best_checkpoint(model, eval_loss, best_eval_loss, path):
    if eval_loss >= best_eval_loss:
        return best_eval_loss

    torch.save(model.state_dict(), path)
    print(f"Saved best checkpoint to {path} | Eval Loss: {eval_loss:.4f}")
    return eval_loss


def evaluate(model, eval_loader, device, max_batches=None):
    model.eval()
    total_eval_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)

            seq_len = input_ids.size(1)
            valid_len = seq_len - 1

            logits = outputs[:, :valid_len, :].contiguous().view(-1, outputs.size(-1))
            shifted_labels = labels[:, :valid_len].contiguous().view(-1)

            loss = compute_kl_div_loss(logits, shifted_labels)
            total_eval_loss += loss.item()
            num_batches += 1

    model.train()
    return total_eval_loss / max(num_batches, 1)


def sample_next_token(logits, temperature=0.8, top_k=50):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, top_k).values[:, -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(model, tokenizer, prompt, device, max_new_tokens=80, context_length=128):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -context_length:]
            logits = model(context)
            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(next_token_logits)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    model.train()
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# 6. Training + evaluation loop
best_eval_loss = float("inf")
model.train()
for epoch in range(args.epoch):
    model.train()
    total_train_loss = 0
    running_train_loss = 0

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids)

        seq_len = input_ids.size(1)
        valid_len = seq_len - 1

        logits = outputs[:, :valid_len, :].contiguous().view(-1, outputs.size(-1))
        shifted_labels = labels[:, :valid_len].contiguous().view(-1)

        loss = compute_kl_div_loss(logits, shifted_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        running_train_loss += loss.item()

        if step % LOG_INTERVAL == 0:
            avg_train_loss = running_train_loss / LOG_INTERVAL
            print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Train Loss: {avg_train_loss:.4f}")
            running_train_loss = 0

        if step % EVAL_INTERVAL == 0:
            eval_loss = evaluate(model, eval_loader, device, max_batches=EVAL_MAX_BATCHES)
            print(
                f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | "
                f"Eval Loss: {eval_loss:.4f} ({EVAL_MAX_BATCHES} batches)"
            )
            best_eval_loss = save_best_checkpoint(
                model,
                eval_loss,
                best_eval_loss,
                BEST_CHECKPOINT_PATH,
            )

        if step % GENERATE_INTERVAL == 0:
            sample = generate_text(
                model,
                tokenizer,
                GENERATE_PROMPT,
                device,
                max_new_tokens=GENERATE_MAX_NEW_TOKENS,
            )
            print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Sample:")
            print(sample)
            print("-" * 80)

    # Evaluation step
    eval_loss = evaluate(model, eval_loader, device)

    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(train_loader):.4f} | Eval Loss: {eval_loss:.4f}")
    best_eval_loss = save_best_checkpoint(
        model,
        eval_loss,
        best_eval_loss,
        BEST_CHECKPOINT_PATH,
    )

# Save model
torch.save(model.state_dict(), FINAL_CHECKPOINT_PATH)
print(f"Saved final checkpoint to {FINAL_CHECKPOINT_PATH}")
print(f"Best eval loss: {best_eval_loss:.4f} | Best checkpoint: {BEST_CHECKPOINT_PATH}")
print("Training complete.")
