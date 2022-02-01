import numpy as np
import torch

from typing import Iterable


class DefaultCollator:
    def __init__(
        self,
        tokenizer=None,
        max_input_length: int = None,
        max_output_length: int = None,
        padding: bool = True,
        pad_id: int = None,
    ):
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.padding = padding
        self.pad_id = pad_id

        if tokenizer is not None:
            self.set_tokenizer(tokenizer)
        else:
            self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        if self.padding and self.pad_id is None:
            self.pad_id = tokenizer.stoi[tokenizer.pad_token]

        self.tokenizer = tokenizer

    def collate_fn(self, batch: Iterable[dict]):
        """
        A model specific collate function that will be passed to the datamodule i.e. the dataloaders.

        Parameters
        ----------
        batch : Is an iterable of elements returned by the getter of a training dataset (an
        instance of torch.utils.data.Dataset)

        Returns
        -------
        The input samples processed to the suitable format that will be passed to the forward method
        of the model by the dataloader.
        """

        collated_batch = {}

        key = "id"
        collated_batch["id"] = torch.tensor([sample[key] for sample in batch], dtype=torch.long)

        key = "text"
        raw_inputs = [sample[key] for sample in batch]
        input_ids = self.tokenizer.tokenize_and_encode_batch(raw_inputs)
        if self.max_input_length is not None:
            input_ids = [i[: self.max_input_length] for i in input_ids]
        if self.padding:
            input_ids = self.pad(input_ids)
        collated_batch["input_ids"] = torch.tensor(input_ids, dtype=torch.long)

        return collated_batch

    def pad(self, data: Iterable[np.array]):
        max_len = max([len(arr) for arr in data])
        padded = np.array(
            [np.pad(arr, (0, max_len - len(arr)), "constant", constant_values=self.pad_id) for arr in data]
        )

        return padded
