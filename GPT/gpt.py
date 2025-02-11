# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
BATCH_SIZE = 32
BLOCK_SIZE = 8
LEARNING_RATE = 1e-2
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#reproducibility
torch.manual_seed(1337)

# data processing
class Data:
    def __init__(self, filename):
        self.read_data(filename)
        self.make_mapping()
        self.make_folds(self.text)

    def read_data(self, filename):
        with open(filename, "r") as f:
            self.text = f.read()

    def make_mapping(self):
        unique_chars = sorted(list(set(self.text)))
        self.stoi = {c: idx for idx, c in enumerate(unique_chars)}
        self.itos = {idx: c for idx, c in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, s: str):
        return list(map(lambda c: self.stoi[c], s))

    def decode(self, ids: list[int]):
        return "".join(list(map(lambda idx: self.itos[idx], ids)))

    def make_folds(self, text: str):
        data = torch.tensor(self.encode(text), dtype=torch.long)
        train_size = int(len(data) * 0.9)
        self.train_data = data[:train_size]
        self.val_data = data[train_size:]

    def get_train_fold(self):
        return self.train_data

    def get_test_fold(self):
        return self.test_data

    def get_vocab_size(self):
        return self.vocab_size

    def get_batch(self, split: str, batch_size=BATCH_SIZE) -> torch.tensor:
        """
        gets a random batch from data, (T, B) shape
        """
        assert(split in {"train", "val"})
        fold = self.train_data if split == "train" else self.val_data
        idxs = torch.randint(len(fold) - BLOCK_SIZE, (batch_size,))
        x = torch.stack([fold[idx:idx+BLOCK_SIZE] for idx in idxs])
        y = torch.stack([fold[idx+1:idx+BLOCK_SIZE+1] for idx in idxs])
        return x, y


# estimate loss
@torch.no_grad()
def estimate_loss(model, data):
    out = dict()
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = data.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# model
class Head(nn.Module):
    """
    one head of self-attention
    """
    def __init__(self, head_size):
        raise NotImplementedError()

    def forward(self, x):
        """
        input of shape (B, T, C)
        output of shape (B, T, head_size)
        """
        raise NotImplementedError()


class MultiHeadAttention(nn.Module):
    """
    multiple heads of self-attention in parallel
    """
    def __init__(self, n_heads, head_size):
        raise NotImplementedError()

    def forward(self, x):
        """
        concats every head
        """
        raise NotImplementedError()


class FeedForward(nn.Module):
    def __init__(self, model_dim):
        raise NotImplementedError()

    def forward(self, input):
        """
        a linear layer with nonlinearity after
        """
        raise NotImplementedError()


class Block(nn.Module):
    def __init__(self, model_dim, n_heads):
        raise NotImplementedError()

    def forward(self, input):
        raise NotImplementedError()


class GPTLanguageModel():
    def __init__(self, vocab_size, model_dim, n_heads):
        raise NotImplementedError()

    def _init_weights(self, module):
        raise NotImplementedError()

    def forward(self, idx, targets=None):
        raise NotImplementedError()

    def generate(self, idx, max_new_tokens):
        raise NotImplementedError()


# train
def train(model, data, iters=MAX_ITERS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(iters):
        if step % EVAL_INTERVAL == 0 or step == iters - 1:
            losses = estimate_loss(model, data)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = data.get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("train loss: ", loss.item())

# generate
def generate(model, data, max_new_tokens=32, batch_size=BATCH_SIZE):
    xb, yb = data.get_batch("train", batch_size=batch_size)
    res = model.generate(xb, max_new_tokens)
    [print(f"batch idx: {idx}, output:\n{data.decode(res[idx].tolist())}") for idx in range(res.shape[0])]

if __name__ == "__main__":
    data = Data("input.txt")
    model = BigramLanguageModel(vocab_size=data.get_vocab_size())
    train(model, data, iters=MAX_ITERS)
    generate(model, data, batch_size=1)
