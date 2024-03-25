import torch


class Dataloader:

    def __init__(self, block_size, batch_size, train_size=0.9):
        self.block_size = block_size
        self.batch_size = batch_size

        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        # encoder: take a string, output a list of integers
        self.encode = lambda s: [stoi[c] for c in s]

        # decoder: take a list of integers, output a string
        self.decode = lambda l: "".join([itos[i] for i in l])

        data = torch.tensor(self.encode(text), dtype=torch.long)

        n = int(train_size * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y
