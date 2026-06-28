import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    A small, pure PyTorch Mamba-style block.

    This keeps the key pieces of Mamba without depending on the fused CUDA
    selective-scan kernel: causal depthwise convolution, input-dependent SSM
    parameters, a diagonal state transition, and an output gate.
    """
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 64,
        d_conv: int = 4,
        dropout: float = 0.1,
        dt_rank: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or max(1, d_model // 16)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)

        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=0,
            groups=d_inner,
        )

        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, x, dt, B, C):
        # x: [batch, seq_len, d_inner]
        # dt: [batch, seq_len, d_inner]
        # B/C: [batch, seq_len, d_state]
        batch_size, seq_len, _ = x.shape
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        D = self.D.float()
        state = x.new_zeros(batch_size, self.d_inner, self.d_state)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t].float()
            dt_t = dt[:, t].float()
            B_t = B[:, t].float()
            C_t = C[:, t].float()

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB_x = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)
            state = dA * state + dB_x

            y_t = torch.sum(state * C_t.unsqueeze(1), dim=-1)
            y_t = y_t + D.unsqueeze(0) * x_t
            outputs.append(y_t.to(dtype=x.dtype))

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = F.pad(x, (self.d_conv - 1, 0))
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = F.silu(x)

        ssm_params = self.x_proj(x)
        dt, B, C = torch.split(ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        x = self.selective_scan(x, dt, B, C)
        x = x * F.silu(gate)
        x = self.out_proj(x)
        return self.dropout(x)


class Mamba(nn.Module):
    """
    A minimal Mamba-style language model: embedding + Mamba blocks + LM head.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 num_blocks: int = 3,
                 d_inner: int = 2048,
                 d_state: int = 64,
                 d_conv: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = x + layer(x)

        x = self.ln(x)
        return self.lm_head(x)
