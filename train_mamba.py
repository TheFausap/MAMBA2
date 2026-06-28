from pathlib import Path

import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from mamba_model import Mamba


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=3, help="number of epochs to train in non-streaming mode")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--block", type=int, default=3)
    parser.add_argument("--state", type=int, default=64)
    parser.add_argument("--conv", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tok_max_len", type=int, default=128, help="max token length or context length if streaming is on")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/dclm-edu")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_tokens", type=int, default=1_000_000)
    parser.add_argument("--eval_examples", type=int, default=1_000)
    parser.add_argument("--eval_max_batches", type=int, default=100)
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--generate_interval", type=int, default=1000)
    parser.add_argument("--generate_max_new_tokens", type=int, default=80)
    parser.add_argument("--generate_prompt", type=str, default="Once upon a time")
    parser.add_argument("--best_checkpoint_path", type=str, default="mamba_best.pt")
    parser.add_argument("--final_checkpoint_path", type=str, default="mamba_final.pt")
    parser.add_argument("--tokenized_dataset_dir", type=str, default=None)

    return parser.parse_args()


class PackedTextDataset(IterableDataset):
    def __init__(self, examples, tokenizer, text_column, sequence_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.sequence_length = sequence_length
        self.tokens_per_chunk = sequence_length + 1
        self.eos_token_id = tokenizer.eos_token_id

    def __iter__(self):
        token_buffer = []

        for example in self.examples:
            if self.text_column not in example:
                available_columns = ", ".join(example.keys())
                raise KeyError(
                    f"Column {self.text_column!r} was not found. Available columns: {available_columns}"
                )

            text = example[self.text_column]
            if text is None:
                continue

            token_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
            if token_ids:
                token_buffer.extend(token_ids)
            token_buffer.append(self.eos_token_id)

            while len(token_buffer) >= self.tokens_per_chunk:
                chunk = token_buffer[: self.tokens_per_chunk]
                del token_buffer[: self.tokens_per_chunk]

                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


def tokenize_padded_example(example, tokenizer, text_column, max_length):
    if text_column not in example:
        available_columns = ", ".join(example.keys())
        raise KeyError(f"Column {text_column!r} was not found. Available columns: {available_columns}")

    encoded = tokenizer(
        str(example[text_column]),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    labels = input_ids.clone()
    labels[:-1] = input_ids[1:]
    labels[-1] = -100
    pad_target_positions = torch.zeros_like(attention_mask, dtype=torch.bool)
    pad_target_positions[:-1] = attention_mask[1:] == 0
    labels[pad_target_positions] = -100
    return {"input_ids": input_ids, "labels": labels}


def build_streaming_loaders(args, tokenizer):
    if args.max_tokens <= 0:
        raise ValueError("--max_tokens must be positive in streaming mode")

    if args.eval_examples > 0:
        eval_stream = load_dataset(args.dataset, split=args.split, streaming=True).take(args.eval_examples)
        eval_dataset = PackedTextDataset(eval_stream, tokenizer, args.text_column, args.tok_max_len)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        eval_loader = None

    train_stream = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.eval_examples > 0:
        train_stream = train_stream.skip(args.eval_examples)
    if args.shuffle_buffer > 0:
        train_stream = train_stream.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    train_dataset = PackedTextDataset(train_stream, tokenizer, args.text_column, args.tok_max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, eval_loader


def build_cached_loaders(args, tokenizer):
    dataset_cache_dir = args.tokenized_dataset_dir
    if dataset_cache_dir is None:
        safe_dataset_name = args.dataset.replace("/", "_")
        dataset_cache_dir = f"{safe_dataset_name}_gpt2_tokenized_{args.tok_max_len}"

    tokenized_dataset_dir = Path(dataset_cache_dir)

    if tokenized_dataset_dir.exists():
        print(f"Loading tokenized dataset from {tokenized_dataset_dir}...")
        dataset = load_from_disk(str(tokenized_dataset_dir))
    else:
        print(f"Loading dataset {args.dataset!r} from Hugging Face...")
        dataset = load_dataset(args.dataset, split=args.split)
        print(f"Tokenizing dataset and saving to {tokenized_dataset_dir}...")
        dataset = dataset.map(
            lambda example: tokenize_padded_example(example, tokenizer, args.text_column, args.tok_max_len),
            batched=False,
            remove_columns=dataset.column_names,
        )
        dataset.save_to_disk(str(tokenized_dataset_dir))

    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    split_index = int(len(dataset) * 0.8)
    train_dataset = dataset.select(range(split_index))
    eval_dataset = dataset.select(range(split_index, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, eval_loader


def compute_loss(logits, labels):
    if not (labels != -100).any():
        return logits.sum() * 0
    return F.cross_entropy(logits, labels, ignore_index=-100)


def forward_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids)
    logits = outputs.contiguous().view(-1, outputs.size(-1))
    shifted_labels = labels.contiguous().view(-1)
    loss = compute_loss(logits, shifted_labels)
    valid_tokens = int((shifted_labels != -100).sum().item())
    return loss, valid_tokens


def save_best_checkpoint(model, eval_loss, best_eval_loss, path):
    if eval_loss >= best_eval_loss:
        return best_eval_loss

    torch.save(model.state_dict(), path)
    print(f"Saved best checkpoint to {path} | Eval Loss: {eval_loss:.4f}")
    return eval_loss


def evaluate(model, eval_loader, device, max_batches=None):
    if eval_loader is None:
        return None

    model.eval()
    total_eval_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            loss, _ = forward_loss(model, batch, device)
            total_eval_loss += loss.item()
            num_batches += 1

    model.train()
    if num_batches == 0:
        return None
    return total_eval_loss / num_batches


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


def print_eval_result(step, eval_loss, max_batches):
    if eval_loss is None:
        print(f"Step {step} | Eval skipped: no eval batches were produced")
        return

    print(f"Step {step} | Eval Loss: {eval_loss:.4f} ({max_batches} max batches)")


def train_streaming(args, model, optimizer, train_loader, eval_loader, tokenizer, device):
    best_eval_loss = float("inf")
    consumed_tokens = 0
    running_loss = 0
    running_batches = 0
    step = 0
    model.train()

    for step, batch in enumerate(train_loader, start=1):
        loss, valid_tokens = forward_loss(model, batch, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        consumed_tokens += valid_tokens
        running_loss += loss.item()
        running_batches += 1

        if step % args.log_interval == 0:
            avg_loss = running_loss / max(running_batches, 1)
            print(
                f"Step {step} | Tokens {consumed_tokens}/{args.max_tokens} | "
                f"Train Loss: {avg_loss:.4f}"
            )
            running_loss = 0
            running_batches = 0

        if eval_loader is not None and step % args.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, device, max_batches=args.eval_max_batches)
            print_eval_result(step, eval_loss, args.eval_max_batches)
            if eval_loss is not None:
                best_eval_loss = save_best_checkpoint(
                    model,
                    eval_loss,
                    best_eval_loss,
                    args.best_checkpoint_path,
                )

        if step % args.generate_interval == 0:
            sample = generate_text(
                model,
                tokenizer,
                args.generate_prompt,
                device,
                max_new_tokens=args.generate_max_new_tokens,
                context_length=args.tok_max_len,
            )
            print(f"Step {step} | Sample:")
            print(sample)
            print("-" * 80)

        if consumed_tokens >= args.max_tokens:
            break

    eval_loss = evaluate(model, eval_loader, device, max_batches=args.eval_max_batches)
    print_eval_result(step, eval_loss, args.eval_max_batches)
    if eval_loss is not None:
        best_eval_loss = save_best_checkpoint(model, eval_loss, best_eval_loss, args.best_checkpoint_path)

    return best_eval_loss, consumed_tokens, step


def train_cached(args, model, optimizer, train_loader, eval_loader, tokenizer, device):
    best_eval_loss = float("inf")
    global_step = 0
    consumed_tokens = 0
    model.train()

    for epoch in range(args.epoch):
        model.train()
        total_train_loss = 0
        running_train_loss = 0

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            loss, valid_tokens = forward_loss(model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            consumed_tokens += valid_tokens
            total_train_loss += loss.item()
            running_train_loss += loss.item()

            if step % args.log_interval == 0:
                avg_train_loss = running_train_loss / args.log_interval
                print(
                    f"Epoch {epoch + 1} | Step {step}/{len(train_loader)} | "
                    f"Tokens {consumed_tokens} | Train Loss: {avg_train_loss:.4f}"
                )
                running_train_loss = 0

            if step % args.eval_interval == 0:
                eval_loss = evaluate(model, eval_loader, device, max_batches=args.eval_max_batches)
                print_eval_result(global_step, eval_loss, args.eval_max_batches)
                if eval_loss is not None:
                    best_eval_loss = save_best_checkpoint(
                        model,
                        eval_loss,
                        best_eval_loss,
                        args.best_checkpoint_path,
                    )

            if step % args.generate_interval == 0:
                sample = generate_text(
                    model,
                    tokenizer,
                    args.generate_prompt,
                    device,
                    max_new_tokens=args.generate_max_new_tokens,
                    context_length=args.tok_max_len,
                )
                print(f"Epoch {epoch + 1} | Step {step}/{len(train_loader)} | Sample:")
                print(sample)
                print("-" * 80)

        eval_loss = evaluate(model, eval_loader, device)
        avg_epoch_loss = total_train_loss / max(len(train_loader), 1)
        if eval_loss is None:
            print(f"Epoch {epoch + 1} | Train Loss: {avg_epoch_loss:.4f} | Eval skipped")
        else:
            print(f"Epoch {epoch + 1} | Train Loss: {avg_epoch_loss:.4f} | Eval Loss: {eval_loss:.4f}")
            best_eval_loss = save_best_checkpoint(model, eval_loss, best_eval_loss, args.best_checkpoint_path)

    return best_eval_loss, consumed_tokens, global_step


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.streaming:
        train_loader, eval_loader = build_streaming_loaders(args, tokenizer)
    else:
        train_loader, eval_loader = build_cached_loaders(args, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mamba(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        num_blocks=args.block,
        d_inner=args.d_model * 8,
        d_state=args.state,
        d_conv=args.conv,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.streaming:
        best_eval_loss, consumed_tokens, steps = train_streaming(
            args,
            model,
            optimizer,
            train_loader,
            eval_loader,
            tokenizer,
            device,
        )
    else:
        best_eval_loss, consumed_tokens, steps = train_cached(
            args,
            model,
            optimizer,
            train_loader,
            eval_loader,
            tokenizer,
            device,
        )

    torch.save(model.state_dict(), args.final_checkpoint_path)
    print(f"Saved final checkpoint to {args.final_checkpoint_path}")
    best_eval_message = f"{best_eval_loss:.4f}" if best_eval_loss < float("inf") else "n/a"
    print(
        f"Best eval loss: {best_eval_message} | Best checkpoint: {args.best_checkpoint_path} | "
        f"Steps: {steps} | Train tokens: {consumed_tokens}"
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
