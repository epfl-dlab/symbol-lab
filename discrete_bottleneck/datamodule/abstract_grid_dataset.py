from torch.utils.data import Dataset


class AbstractGridDataset(Dataset):
    """
    An abstract class for a torch.utils.data.Dataset of randomly generated grids.
    """

    def __init__(self, split, seed, **kwargs):
        """
        Parameters
        ----------
        split : The split to load. Must be equal to 'train', 'val' or 'test'.
        seed : Random seed

        kwargs: generator function's parameters

        Returns
        -------
        An instance of the Grid dataset that extends torch.utils.data.Dataset.
        """
        super().__init__()
        self.params = kwargs

        assert split in {"train", "val", "test"}, "Unexpected split reference"
        if split == "val":
            seed += 10
        elif split == "test":
            seed += 23
        self.params["seed"] = seed

        self.data = self._generate_data()

    def _generate_data(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"id": self.data[idx][0], "text": self.data[idx][1]}
