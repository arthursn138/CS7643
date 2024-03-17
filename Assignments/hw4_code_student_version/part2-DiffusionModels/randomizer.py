import torch
def DL_random(shape, int_range = None, normal = True, seed=None):
    """Generate a random matrix of the given shape.
    If int_range is provided, the matrix will be filled with integers in the range [int_range[0], int_range[1])
    If int_range is not provided, the matrix will be filled with random floats (uniform or normal distribution)

    * Do not modify this function *
    * Seed parameter is for unit tests and autograder *

    Args:
        shape (tuple): the shape of the matrix to generate
        int_range (tuple): the range of integers to use (optional)
        normal (true): whether to use uniform or normal distribution - default is normal
        seed (int): the seed to use for random number generation
    
    Returns:
        torch.Tensor: the generated matrix

    """
    if seed is not None:
        torch.manual_seed(seed)
    if int_range is not None:
        assert len(int_range) == 2
        return torch.randint(*int_range, shape)
    else:
        if normal:
            return torch.randn(*shape)
        else:
            return torch.rand(*shape)