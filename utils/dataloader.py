import torch
from transformers import AutoTokenizer


class Dataloader:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, block_size, batch_size, train_size=0.9):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.block_size = block_size
        self.batch_size = batch_size

        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"][0, 1:-1]

        unique_ids = sorted(list(set(input_ids.tolist())))
        self.vocab_size = len(unique_ids)

        self.id_to_idx = {id_: i for i, id_ in enumerate(unique_ids)}
        self.idx_to_id = {i: id_ for i, id_ in enumerate(unique_ids)}

        # encoder: take a string, output a list of integers
        self.encode = lambda s: [
            self.id_to_idx[id_]
            for id_ in self.tokenizer(s, add_special_tokens=False)["input_ids"]
        ]

        # decoder: take a list of integers, output a string
        self.decode = lambda l: self.tokenizer.decode([self.idx_to_id[i] for i in l])

        input_ids = torch.tensor(self.encode(text), dtype=torch.long)

        n = int(train_size * input_ids.shape[0])
        self.train_data = input_ids[:n]
        self.val_data = input_ids[n:]

    def get_batch(self, split, random=True):
        data = self.train_data if split == "train" else self.val_data
        if random:
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        else:
            ix = torch.arange(0, len(data) - self.block_size)

        x = torch.stack([data[i : i + self.block_size + 1] for i in ix])
        return x
