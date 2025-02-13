# imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import typing as tp

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

    def decode(self, ids: tp.List[int]):
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
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.query_embed = nn.Linear(embed_size, head_size, bias=False)
        self.key_embed = nn.Linear(embed_size, head_size, bias=False)
        self.value_embed = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((BLOCK_SIZE, BLOCK_SIZE))))
        self.embed_size = embed_size

    def forward(self, x):
        """
        input of shape (B, T, C)
        output of shape (B, T, head_size)
        """
        B, T, C = x.shape
        #breakpoint()
        queries = self.query_embed(x)
        keys = self.key_embed(x)
        values = self.value_embed(x)
        weights = queries @ keys.permute(0, 2, 1) / (self.embed_size ** 0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)

        return weights @ values


class MultiHeadAttention(nn.Module):
    """
    multiple heads of self-attention in parallel
    """
    def __init__(self, n_heads, embed_size, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_size, head_size) for n in range(n_heads)])
        self.linear = nn.Linear(n_heads * head_size, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        concats every head
        """
        cat = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.relu(self.linear(cat))


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.ff = nn.Linear(embed_size, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        a linear layer with nonlinearity after
        """
        return self.relu(self.ff(x))


class Block(nn.Module):
    def __init__(self, n_heads, embed_size, head_size):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, embed_size, head_size)
        self.ff = FeedForward(embed_size)

    def forward(self, x):
        heads_out = self.attention(x)
        return self.ff(heads_out)


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, n_heads, head_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention_block = Block(n_heads, embed_size, head_size)
        self.head = nn.Linear(embed_size, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        embeds = self.embedding(idx)
        embeds = self.attention_block(embeds)
        logits = self.head(embeds)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, _ = self.forward(idx[:,-BLOCK_SIZE:,])
            #breakpoint()
            last_t = logits[:, -1, :]
            probs = F.softmax(last_t, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            #breakpoint()
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx, _


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
    res, _ = model.generate(xb, max_new_tokens)
    [print(f"batch idx: {idx}, output:\n{data.decode(res[idx].tolist())}") for idx in range(res.shape[0])]


def test_module():
    data = Data("input.txt")
    model = GPTLanguageModel(vocab_size=data.get_vocab_size(), n_heads=8, embed_size=16, head_size=4)
    # x = torch.as_tensor(data.get_batch("train", batch_size=1)[0].unsqueeze(2), dtype=torch.float)
    x = data.get_batch("train", batch_size=1)[0]
    res = model.generate(x, max_new_tokens=32)
    res_decoded = data.decode(res[0].tolist())

    # breakpoint()
    return res_decoded


def main():
    data = Data("input.txt")
    model = GPTLanguageModel(vocab_size=data.get_vocab_size(), n_heads=8, embed_size=16, head_size=4)
    train(model, data, iters=MAX_ITERS)
    generate(model, data, batch_size=1)


if __name__ == "__main__":
    #print(test_module())
    main()
