# anything thats in Pytorch forecasting utils should go here. i want it all locally redone
import torch

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


def masked_op(
    tensor: torch.Tensor, op: str = "mean", dim: int = 0, mask: torch.Tensor = None
) -> torch.Tensor:
    """Calculate operation on masked tensor.

    Args:
        tensor (torch.Tensor): tensor to conduct operation over
        op (str): operation to apply. One of ["mean", "sum"]. Defaults to "mean".
        dim (int, optional): dimension to average over. Defaults to 0.
        mask (torch.Tensor, optional): boolean mask to apply (True=will take mean, False=ignore).
            Masks nan values by default.

    Returns:
        torch.Tensor: tensor with averaged out dimension
    """
    if mask is None:
        mask = ~torch.isnan(tensor)
    masked = tensor.masked_fill(~mask, 0.0)
    summed = masked.sum(dim=dim)
    if op == "mean":
        return summed / mask.sum(dim=dim)  # Find the average
    elif op == "sum":
        return summed
    else:
        raise ValueError(f"unkown operation {op}")
