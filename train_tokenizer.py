import os
import time

from utils.basic import BasicTokenizer


if __name__ == "__main__":
    # open some text and train a vocab of 512 tokens
    text = open("shakespeare.txt", "r", encoding="utf-8").read()

    # create a directory for models, so we don't pollute the current directory
    os.makedirs("models", exist_ok=True)

    t0 = time.time()

    # construct the Tokenizer object and kick off verbose training
    tokenizer = BasicTokenizer()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    tokenizer.save("tokenizer")

    t1 = time.time()

    print(f"Training took {t1 - t0:.2f} seconds")
