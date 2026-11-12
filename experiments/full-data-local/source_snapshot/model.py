"""Small causal byte transformer: a runnable method implementation, not GPT-4."""
from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn

PAD, BOS, SEP = 0, 257, 258
VOCAB = 259


@dataclass
class ModelConfig:
    width: int = 64
    layers: int = 2
    heads: int = 4
    context: int = 384
    classes: int = 3


def device_for(name='auto'):
    if name == 'auto':
        name = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name not in ('mps', 'cpu'):
        raise ValueError('device must be auto, mps or cpu')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable. Use --device cpu, native arm64 Python, or a supported macOS/PyTorch version.')
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def encode(problem, steps, context):
    """Keep the problem head plus the newest solution bytes within a hard bound."""
    if context < 32:
        raise ValueError('context must be >= 32')
    question = [b+1 for b in problem.encode('utf-8')][:context // 3]
    solution = [b+1 for b in '\n'.join(steps).encode('utf-8')]
    prefix = [BOS] + question + [SEP]
    capacity = context - len(prefix)
    truncated = len(solution) > capacity
    return prefix + solution[-capacity:], len(prefix), truncated


def batch_tokens(sequences, device):
    lengths = torch.tensor([len(x) for x in sequences], device=device)
    tokens = torch.zeros((len(sequences), int(lengths.max())), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return tokens, lengths


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.width % config.heads or config.layers < 1:
            raise ValueError('width must be divisible by heads and layers positive')
        self.config = config
        self.token = nn.Embedding(VOCAB, config.width, padding_idx=PAD)
        self.position = nn.Embedding(config.context, config.width)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, config.width * 4,
                                           dropout=0.0, batch_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        # TransformerEncoder clones initially identical layers; independently initialize.
        for block in self.blocks.layers:
            for parameter in block.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        # Initialize the classifier last so PRM/ORM share identical seeded trunks.
        self.head = nn.Linear(config.width, config.classes)

    def forward(self, tokens):
        length = tokens.shape[1]
        hidden = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        causal = torch.ones((length, length), dtype=torch.bool, device=tokens.device).triu(1)
        # Right padding cannot affect preceding tokens under causal attention.
        return self.head(self.norm(self.blocks(hidden, mask=causal)))


def save_model(path, model, metadata):
    torch.save({'config': asdict(model.config), 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'metadata': metadata}, path)


def load_model(path, device='auto'):
    device = device_for(device)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = RewardModel(ModelConfig(**payload['config']))
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval(), payload['metadata']
