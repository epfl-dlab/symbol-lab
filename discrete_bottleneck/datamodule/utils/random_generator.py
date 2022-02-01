import numpy as np


def random_grid(num_rows: int, num_cols: int, num_object_classes: int, num_objects_to_place: int):
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
    num_objects_to_place : The number of objects to place on the grid

    Returns
    -------
    A string representation of the generated grid.

    """
    assert num_objects_to_place <= num_rows * num_cols, "The number of objects on the grid is larger than the grid"
    assert num_rows >= 1, "The number of rows on the grid must be larger bigger or equal to 1"
    assert num_cols >= 1, "The number of cols on the grid must be larger bigger or equal to 1"
    assert num_object_classes >= 1, "The number of object classes must be larger bigger or equal to 1"

    grid = np.zeros((num_rows, num_cols), dtype=int)

    c = 0
    objects_to_place = np.random.choice(num_object_classes, num_objects_to_place) + 1  # object_ids start from 1
    while c != num_objects_to_place:
        row = np.random.choice(num_rows, 1)[0]
        col = np.random.choice(num_cols, 1)[0]

        if grid[row, col] != 0:
            continue

        grid[row, col] = objects_to_place[c]
        c += 1

    return str(grid.flatten()).strip("[]")


def random_grid_dataset(
    num_samples: int, num_rows: int, num_cols: int, num_object_classes: int, num_objects_to_place: int, seed: int = None
):
    """
    Generates a random dataset of grids in which all of the grids have the same size and number of objects.

    Parameters
    ----------
    num_samples : The number of samples in the dataset
    num_rows : The number of rows in the grid
    num_cols : The number of columns in the grid
    num_object_classes: The number of different object classes (types) considered
    num_objects_to_place : The number of objects to place on the grid
    seed : Random seed

    Returns
    -------
    A list of strings, where each string corresponds to a random grid
    """
    if seed is not None:
        np.random.seed(seed)

    dataset = []

    for sample_id in range(num_samples):
        grid = random_grid(
            num_rows=num_rows,
            num_cols=num_cols,
            num_object_classes=num_object_classes,
            num_objects_to_place=num_objects_to_place,
        )

        dataset.append(grid)

    return dataset
