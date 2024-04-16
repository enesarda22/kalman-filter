import os

import torch

from utils.basic import BasicTokenizer


class Dataloader:

    def __init__(self, block_size, batch_size, train_size=0.9):
        self.block_size = block_size
        self.batch_size = batch_size

        self.tokenizer = BasicTokenizer()
        self.tokenizer.load("tokenizer.model")

        self.vocab_size = len(self.tokenizer.vocab)

        if not os.path.isfile("data.pt"):
            with open("shakespeare.txt", "r", encoding="utf-8") as f:
                text = f.read()

            tokens = self.tokenizer.encode(text)

            data = torch.tensor(tokens, dtype=torch.long)
            torch.save(data, "data.pt")
        else:
            data = torch.load("data.pt")

        n = int(train_size * data.shape[0])
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split, random=True):
        data = self.train_data if split == "train" else self.val_data
        if random:
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        else:
            ix = torch.arange(0, len(data) - self.block_size)

        x = torch.stack([data[i : i + self.block_size + 1] for i in ix])
        return x
