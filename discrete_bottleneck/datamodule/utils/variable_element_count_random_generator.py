import numpy as np
from .random_generator import random_grid


def variable_element_count_random_grid(
    num_rows: int, num_cols: int, num_object_classes: int, min_num_objects_to_place: int, max_num_objects_to_place: int, **kwargs
):
    """
    This function randomly generates a single 2-D grid.
    Objects are represented with their corresponding class ID (\\in [1, num_object_classes])
    Empty cells are denoted with 0.

    Example input/output:
        input: (2, 3, 2, 3)

        output: "0 0 1 0 1 3"

        corresponding to grid:
            0 0 1
            0 1 2

    Parameters
    ----------
    num_rows : The number of rows in the grid
    num_cols : The number of columns in the grid
    num_object_classes: The number of different object classes (types) considered
    min_num_objects_to_place : The minimum number of objects to place on the grid
    max_num_objects_to_place : The maximum number of objects to place on the grid

    Returns
    -------
    A string representation of the generated grid.

    """
    assert min_num_objects_to_place >= 0, "The minimum number of objects must be positive"
    assert (
        max_num_objects_to_place <= num_rows * num_cols
    ), "The maximum number of objects on the grid cannot be larger than the grid"

    num_objects_to_place = np.random.choice(range(min_num_objects_to_place, max_num_objects_to_place + 1), 1)
    return random_grid(
        num_rows=num_rows,
        num_cols=num_cols,
        num_object_classes=num_object_classes,
        num_objects_to_place=num_objects_to_place,
    )


def variable_element_count_random_grid_dataset(
    num_samples: int,
    num_rows: int,
    num_cols: int,
    num_object_classes: int,
    min_num_objects_to_place: int,
    max_num_objects_to_place: int,
    seed: int = None,
    **kwargs
):
    """
    Generates a random dataset of grids in which all of the grids have the same size and number of objects.

    Parameters
    ----------
    num_samples : The number of samples in the dataset
    num_rows : The number of rows in the grid
    num_cols : The number of columns in the grid
    num_object_classes: The number of different object classes (types) considered
    min_num_objects_to_place : The minimum number of objects to place on the grid
    max_num_objects_to_place : The maximum number of objects to place on the grid
    seed : Random seed

    Returns
    -------
    A list of strings, where each string corresponds to a random grid
    """
    if seed is not None:
        np.random.seed(seed)

    dataset = []

    for sample_id in range(num_samples):
        grid = variable_element_count_random_grid(
            num_rows=num_rows,
            num_cols=num_cols,
            num_object_classes=num_object_classes,
            min_num_objects_to_place=min_num_objects_to_place,
            max_num_objects_to_place=max_num_objects_to_place,
        )

        dataset.append(grid)

    return dataset
