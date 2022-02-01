import numpy as np

from typing import Iterable
from discrete_bottleneck.datamodule.abstract_grid_dataset import AbstractGridDataset


class SimpleTokenizer:
    bos_token = "BOS"
    eos_token = "EOS"
    pad_token = "PAD"

    def __init__(self, special_tokens=[bos_token, eos_token, pad_token]):
        self.special_tokens = special_tokens
        self.trained = False
        self.stoi = {}
        self.itos = {}

    def train_tokenizer(self, dataset: AbstractGridDataset):
        stoi = self.stoi
        stoi["0"] = 0  # empty

        num_object_classes = dataset.params["num_object_classes"]
        for i in range(num_object_classes):
            stoi[str(i + 1)] = i + 1

        for st in self.special_tokens:
            stoi[st] = len(stoi)

        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
        self.trained = True

    @staticmethod
    def tokenize(text: str):
        return text.split()

    @staticmethod
    def tokenize_batch(data: Iterable[str]):
        return [SimpleTokenizer.tokenize(text) for text in data]

    def tokenize_and_encode(self, text: str, add_bos_token=True, add_eos_token=True):
        return self.encode(self.tokenize(text), add_bos_token=add_bos_token, add_eos_token=add_eos_token)

    def tokenize_and_encode_batch(self, data: Iterable[str], add_bos_token=True, add_eos_token=True):
        return [
            self.tokenize_and_encode(text, add_bos_token=add_bos_token, add_eos_token=add_eos_token) for text in data
        ]

    def encode(self, l: Iterable[str], add_bos_token=True, add_eos_token=True):
        if add_bos_token and add_eos_token:
            encoded_seq = np.zeros(len(l) + 2, dtype=np.int32)
            encoded_seq[0] = self.stoi[self.bos_token]
            encoded_seq[-1] = self.stoi[self.eos_token]
            idx = 1
        elif add_bos_token:
            encoded_seq = np.zeros(len(l) + 1, dtype=np.int32)
            encoded_seq[0] = self.stoi[self.bos_token]
            idx = 1
        elif add_eos_token:
            encoded_seq = np.zeros(len(l) + 1, dtype=np.int32)
            encoded_seq[-1] = self.stoi[self.eos_token]
            idx = 0
        else:
            encoded_seq = np.zeros(len(l) + 1, dtype=np.int32)
            idx = 0

        for token in l:
            encoded_seq[idx] = self.stoi[token]
            idx += 1

        return encoded_seq

    def encode_batch(self, data: Iterable[Iterable[str]], add_bos_token=True, add_eos_token=True):
        return [self.encode(l, add_bos_token, add_eos_token) for l in data]

    def decode(self, l: Iterable[int], strip_pad_tokens=True, strip_bos_token=True, strip_eos_token=True):
        to_ignore = set()
        if strip_pad_tokens:
            to_ignore.add(self.stoi[self.pad_token])

        if strip_bos_token:
            to_ignore.add(self.stoi[self.bos_token])

        if strip_eos_token:
            to_ignore.add(self.stoi[self.eos_token])

        return " ".join([self.itos[idx] for idx in l if idx not in to_ignore])

    def decode_batch(
        self, data: Iterable[Iterable[int]], strip_pad_tokens=True, strip_bos_token=True, strip_eos_token=True
    ):
        return [self.decode(l, strip_pad_tokens, strip_bos_token, strip_eos_token) for l in data]

    def dump(self, path):
        # TODO
        raise NotImplementedError()

    def load(self, path):
        # TODO
        raise NotImplementedError()
