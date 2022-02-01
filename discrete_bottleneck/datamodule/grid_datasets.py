from torch.utils.data import Dataset
from .utils.random_generator import random_grid_dataset
from .utils.variable_element_count_random_generator import variable_element_count_random_grid_dataset
from .abstract_grid_dataset import AbstractGridDataset


class ConstantGridDataset(AbstractGridDataset):
    """
    A torch.utils.data.Dataset of randomly generated grids of equal size and number of objects.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        split : The split to load. Must be equal to 'train', 'val' or 'test'.
        seed : Random seed.

        num_samples : The number of samples in the dataset
        num_rows : The number of rows in the grid
        num_cols : The number of columns in the grid
        num_object_classes: The number of different object classes (types) considered
        num_objects_to_place : The number of objects to place on the grid


        Returns
        -------
        An instance of the Grid dataset that extends torch.utils.data.Dataset.
        """

        super().__init__(**kwargs)

    def _generate_data(self):
        """
        random_grid_dataset returns a list of sampled grids such as the following:
            ["0 0 1 0 1 3", "1 0 1 0 2 1", "1 3 2 0 1 2", ...]
        This function associates an id to each text in the list above and returns a list of tuples:
            [(0, "0 0 1 0 1 3"), (1, "1 0 1 0 2 1"), (2, "1 3 2 0 1 2"), ...]
        """
        return list(enumerate(random_grid_dataset(**self.params)))


class VariableElementCountGridDataset(AbstractGridDataset):
    """
    A torch.utils.data.Dataset of randomly generated grids are of equal size, but the number of objects on the grid
    varies.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        split : The split to load. Must be equal to 'train', 'val' or 'test'.
        seed : Random seed.

        num_samples : The number of samples in the dataset
        num_rows : The number of rows in the grid
        num_cols : The number of columns in the grid
        num_object_classes: The number of different object classes (types) considered
        min_num_objects_to_place : The minimum number of objects to place on the grid
        max_num_objects_to_place : The maximum number of objects to place on the grid


        Returns
        -------
        An instance of the Grid dataset that extends torch.utils.data.Dataset.
        """
        super().__init__(**kwargs)

    def _generate_data(self):
        """
        variable_element_count_random_grid_dataset returns a list of sampled grids such as the following:
            ["0 0 1 0 1 3", "1 0 0 0 2 0", "1 3 2 1 1 2", ...]
        This function associates an id to each text in the list above and returns a list of tuples:
            [(0, "0 0 1 0 1 3"), (1, "1 0 0 0 2 0"), (2, "1 3 2 1 1 2"), ...]
        """
        return list(enumerate(variable_element_count_random_grid_dataset(**self.params)))
