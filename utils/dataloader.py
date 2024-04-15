import torch


class Dataloader:

    def __init__(self, block_size, batch_size, train_size=0.9):
        self.block_size = block_size
        self.batch_size = batch_size

        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        chars = ["CLS"] + chars
        self.vocab_size = len(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.cls_idx = stoi["CLS"]

        # encoder: take a string, output a list of integers
        self.encode = lambda s: [stoi[c] for c in s]

        # decoder: take a list of integers, output a string
        self.decode = lambda l: "".join([itos[i] for i in l])

        data = torch.tensor(self.encode(text), dtype=torch.long)

        n = int(train_size * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split, random=True):
        data = self.train_data if split == "train" else self.val_data
        if random:
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        else:
            ix = torch.arange(0, len(data) - self.block_size)

        x = torch.stack([data[i : i + self.block_size] for i in ix])

        pad = self.cls_idx * torch.ones(self.batch_size, 1, dtype=torch.long)
        x = torch.hstack([pad, x])
        return x
