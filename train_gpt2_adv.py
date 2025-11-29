import os
import math
import time
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from torch.backends.cuda import sdp_kernel  # imported for completeness

# -----------------------------------------------------------------------------
# Configuration (training & data hyperparameters)

DATA_ROOT = "edu_fineweb10B"

TOTAL_BATCH_SIZE = 131072        # total batch size in number of tokens (2^17)
MICRO_BATCH_SIZE = 32            # per-GPU micro batch size
SEQ_LEN = 1024                   # sequence length (block size)

MAX_LR = 6e-4
MIN_LR_FACTOR = 0.1              # min_lr = MAX_LR * MIN_LR_FACTOR
WARMUP_STEPS = 750
MAX_STEPS = 19073                # ~1 epoch for 10B tokens with 0.5M token batch

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

VAL_INTERVAL = 250               # validate every N steps
VAL_STEPS = 20                   # how many batches to average for val loss
HELLA_INTERVAL = 250             # HellaSwag eval interval
MAX_HELLASWAG_EXAMPLES = 1000    # number of HellaSwag examples to use
GEN_INTERVAL = 250               # text generation interval
NUM_GEN_RETURN_SEQS = 4
GEN_MAX_LENGTH = 32
GEN_TOP_K = 50

LOG_DIR = "log"
LOG_FILENAME = "log.txt"

GLOBAL_SEED = 1337

# -----------------------------------------------------------------------------
# Model components


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, max_position_embeddings=2048):
        """
        Rotary positional embedding (RoPE).

        Args:
            dim: head dimension (head_dim)
            base: RoPE base (10000 is the standard value)
        """
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_pos, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)       # (max_pos, dim)
        # shape: (1, 1, max_pos, dim) so it can broadcast to (B, n_head, T, dim)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        """
        Apply RoPE to q and k.

        Args:
            q, k: (B, n_head, T, head_dim)
        """
        if seq_len is None:
            seq_len = q.size(-2)
        # It is critical to match the dtype/device of q. Otherwise tensors
        # become fp32 and Flash Attention may be disabled.
        cos = self.cos_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        q2 = (q * cos) + (self._rotate_half(q) * sin)
        k2 = (k * cos) + (self._rotate_half(k) * sin)
        return q2, k2


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # RoPE configuration
        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            rope_base = getattr(config, "rope_base", 10000.0)
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                base=rope_base,
                max_position_embeddings=config.block_size,
            )
        else:
            self.rotary_emb = None

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # apply RoPE to q, k
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, seq_len=T)

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_type = getattr(config, "mlp_type", "gelu")
        self.mlp_type = mlp_type

        if mlp_type == "swiglu":
            # Set inner dim ~ 8/3 * d so that parameter count matches 4d-GELU
            inner_dim = int(4 * config.n_embd * 2 / 3)
            # Round up to multiple of 256 for efficiency
            inner_dim = ((inner_dim + 255) // 256) * 256
            self.inner_dim = inner_dim
            # value and gate
            self.c_fc = nn.Linear(config.n_embd, 2 * inner_dim)
            self.c_proj = nn.Linear(inner_dim, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1
        else:
            # Standard GELU MLP
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.act = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        if self.mlp_type == "swiglu":
            x_in = self.c_fc(x)
            x_gate, x_up = x_in.chunk(2, dim=-1)
            x = F.silu(x_gate) * x_up
            x = self.c_proj(x)
        else:
            x = self.c_fc(x)
            x = self.act(x)
            x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # additional options
    use_rope: bool = True          # use RoPE for fresh training
    rope_base: float = 10000.0
    mlp_type: str = "swiglu"       # can also be "gelu"


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        NormLayer = nn.LayerNorm

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=NormLayer(config.n_embd),
        )
        # If we use RoPE, we don't need absolute position embeddings
        if not getattr(config, "use_rope", True):
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        if getattr(self.config, "use_rope", True):
            # If using RoPE, we skip absolute position embeddings
            x = tok_emb
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
            pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
            x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from Hugging Face."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),      # 1558M
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024   # always 1024 for GPT model checkpoints

        # Configuration compatible with Hugging Face GPT-2 checkpoints
        config_args["use_rope"] = False
        config_args["mlp_type"] = "gelu"

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # drop mask/buffer

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # OpenAI checkpoints use a Conv1D module; here we use Linear,
        # so we must transpose some weights when importing.
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for Conv1D weights that must be transposed
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy for other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(
            torch.optim.AdamW
        ).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer


# -----------------------------------------------------------------------------
# Data loading

import tiktoken
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        shards = os.listdir(DATA_ROOT)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(DATA_ROOT, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (
            B * T * self.num_processes + 1
        ) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# HellaSwag evaluation helper
# takes tokens, mask, and logits, returns the index of the completion
# with the lowest loss


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # shift mask so it starts at last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# -----------------------------------------------------------------------------
# Training loop setup
# simple launch:
#   python train_gpt2_adv.py
# DDP launch for e.g. 8 GPUs:
#   torchrun --standalone --nproc_per_node=8 train_gpt2_adv.py

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP currently assumes CUDA, we set the device according to rank
    assert torch.cuda.is_available(), "DDP currently requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# PyTorch can be particular about device vs device_type
device_type = "cuda" if str(device).startswith("cuda") else "cpu"

torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)

enc = tiktoken.get_encoding("gpt2")

B = MICRO_BATCH_SIZE
T = SEQ_LEN
total_batch_size = TOTAL_BATCH_SIZE

assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)

torch.set_float32_matmul_precision("high")

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)

use_compile = False  # On Windows Triton is not available, so torch.compile cannot be used
if use_compile:
    # dynamic=True supports variable-length sequences for HellaSwag / generation
    model = torch.compile(model, dynamic=True)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the raw unwrapped model

min_lr = MAX_LR * MIN_LR_FACTOR


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    # 2) if it > MAX_STEPS, return min learning rate
    if it > MAX_STEPS:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (MAX_LR - min_lr)


# optimize!
optimizer = raw_model.configure_optimizers(
    weight_decay=WEIGHT_DECAY, learning_rate=MAX_LR, device_type=device_type
)

# create the log directory we will write checkpoints and logs to
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, LOG_FILENAME)
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

start_time = time.time()

# initialize before training loop
best_hellaswag = 0.0

for step in range(MAX_STEPS):
    t0 = time.time()
    last_step = step == MAX_STEPS - 1

    # periodically evaluate validation loss
    if step % VAL_INTERVAL == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(VAL_STEPS):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / VAL_STEPS
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            val = val_loss_accum.item()
            print(f"validation loss: {val:.4f} | elapsed {elapsed_str}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val:.4f} elapsed {elapsed_str}\n")
            if step > 0 and (step % (5_000) == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(LOG_DIR, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # periodically evaluate HellaSwag
    if (step % HELLA_INTERVAL == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # limit the number of evaluated HellaSwag examples
            if i >= MAX_HELLASWAG_EXAMPLES:
                break

            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(
                    device_type=device_type, dtype=torch.bfloat16
                ):
                    logits, _ = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(
                num_correct_norm, dtype=torch.long, device=device
            )
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

            # save checkpoint when we get a new best HellaSwag score
            if acc_norm > best_hellaswag:
                best_hellaswag = acc_norm
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "hellaswag": acc_norm,
                }
                torch.save(checkpoint, os.path.join(LOG_DIR, "model_best.pt"))
                print(
                    f"New best HellaSwag: {acc_norm:.4f}, saved model_best.pt"
                )

    # periodically generate samples (except at step 0, which is mostly noise)
    if ((step > 0 and step % GEN_INTERVAL == 0) or last_step) and (
        not use_compile
    ):
        model.eval()
        num_return_sequences = NUM_GEN_RETURN_SEQS
        max_length = GEN_MAX_LENGTH
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(
                    device_type=device_type, dtype=torch.bfloat16
                ):
                    logits, _ = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling (Hugging Face pipeline default is k=50)
                topk_probs, topk_indices = torch.topk(
                    probs, GEN_TOP_K, dim=-1
                )
                # select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # one optimization step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # this field is also used by the forward pass when using DDP
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == grad_accum_steps - 1
            )
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        # scale the loss to account for gradient accumulation.
        # gradients add on each successive backward() call, corresponding to
        # a SUM in the objective, but we want a MEAN, so we scale here.
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()  # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
