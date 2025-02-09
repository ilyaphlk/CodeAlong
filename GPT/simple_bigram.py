# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32
block_size = 8
vocab_size = None
learning_rate = 1e-2
max_iters = 3000
eval_interval = 300
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#reproducibility
torch.manual_seed(1337)

# data processing
with open("input.txt", "r") as f:
    text = f.read()

unique_chars = sorted(list(set(text)))
stoi = {c: idx for idx, c in enumerate(unique_chars)}
itos = {idx: c for idx, c in enumerate(unique_chars)}
vocab_size = len(unique_chars)

def encode(s):
    return list(map(lambda c: stoi[c], s))

def decode(ids):
    return "".join(list(map(lambda idx: itos[idx], ids)))

data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
val_data = data[train_size:]


# data loading
def get_batch(split: str) -> torch.tensor:
    """
    gets a random batch from data, (T, B) shape
    """
    assert(split in {"train", "val"})
    fold = train_data if split == "train" else val_data
    idxs = torch.randint(len(fold) - block_size, (batch_size,))
    x = torch.stack([fold[idx:idx+block_size] for idx in idxs])
    y = torch.stack([fold[idx+1:idx+block_size+1] for idx in idxs])
    return x, y

# estimate loss

# model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        """
        make layers
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        """
        inputs: (B, T)
        targets: (B, T)
        return outputs (B, T, C) and loss (optional, float)
        """
        loss = None
        logits = self.embed(inputs)  # (B, T, C)
        B, T, C = logits.shape
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        idx = inputs of shape (B, T)
        return outputs of shape (B, T+max_new_tokens)
        """
        for i in range(max_new_tokens):
            logits, _ = self.forward(idx)
            last_t = logits[:,-1,:]  # (B, C)
            probs = F.softmax(last_t, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

# train
m = BigramLanguageModel(vocab_size=vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for step in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("train loss: ", loss.item())

# generate
torch.manual_seed(1337)
xb, yb = get_batch("train")
res = m.generate(xb, max_new_tokens=32)
[print(f"idx: {idx}, output:\n{decode(res[idx].tolist())}") for idx in range(res.shape[0])]
