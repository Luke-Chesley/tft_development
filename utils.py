# anything thats in Pytorch forecasting utils should go here. i want it all locally redone

def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """

    return min(round(1.6 * n**0.56), max_size) if n > 2 else 1
