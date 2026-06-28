import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mamba_model import Mamba


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate text with the TinyStories Mamba checkpoint.")
    parser.add_argument("--checkpoint", default="mamba_tinystories.pt", help="Path to model state_dict.")
    parser.add_argument("--prompt", default="Once upon a time", help="Text prompt to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Use 0 for greedy.")
    parser.add_argument("--top-k", type=int, default=50, help="Keep only the top-k tokens. Use 0 to disable.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold. Use 1.0 to disable.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cpu, or mps.")
    parser.add_argument("--context-length", type=int, default=128, help="Maximum context tokens to feed the model.")

    # Defaults match train_mamba_tinystories.py.
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=None, help="Number of Mamba blocks. Inferred from checkpoint by default.")
    parser.add_argument("--d-inner", type=int, default=512)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def choose_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, context_length, temperature, top_k, top_p, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.numel() == 0:
        input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)

    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -context_length:]
        logits = model(context)

        next_token_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    args = build_arg_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    state_dict = torch.load(args.checkpoint, map_location=device)
    num_blocks = args.num_blocks
    if num_blocks is None:
        num_blocks = 1 + max(
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("layers.")
        )

    model = Mamba(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        num_blocks=num_blocks,
        d_inner=args.d_inner,
        d_state=args.d_state,
        d_conv=args.d_conv,
        dropout=args.dropout,
    ).to(device)

    model.load_state_dict(state_dict)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
