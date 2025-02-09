# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32
block_size = 8
learning_rate = 1e-2
max_iters = 3000
eval_interval = 300
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def get_batch(self, split: str) -> torch.tensor:
        """
        gets a random batch from data, (T, B) shape
        """
        assert(split in {"train", "val"})
        fold = self.train_data if split == "train" else self.val_data
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
def train(model, data, iters=max_iters):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for step in range(iters):
        xb, yb = data.get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("train loss: ", loss.item())

# generate
def generate(model, data, max_new_tokens=32):
    xb, yb = data.get_batch("train")
    res = model.generate(xb, max_new_tokens)
    [print(f"idx: {idx}, output:\n{data.decode(res[idx].tolist())}") for idx in range(res.shape[0])]

if __name__ == "__main__":
    data = Data("input.txt")
    model = BigramLanguageModel(vocab_size=data.get_vocab_size())
    train(model, data, iters=max_iters)
    generate(model, data)
