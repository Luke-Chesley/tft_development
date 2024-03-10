from typing import Dict, Tuple,List,Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.utils import get_embedding_size

class TimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return super().forward(x)
        x_reshape = x.contiguous().view(-1, x.size(1))

        y = super().forward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class MultiEmbedding(nn.Module):
    """
    Embedding layer for categorical variables including groups of categorical variables.

    Possibly could be the static covariate encoder layer, not sure

    """

    def __init__(
        self,
        embedding_sizes: Union[
            Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]
        ],
        x_categoricals: List[str] = None,
        categorical_groups: Dict[str, List[str]] = {},
        embedding_paddings: List[str] = [],
        max_embedding_size: int = None,
    ):
        super().__init__()

        if isinstance(embedding_sizes, dict):
            self.concat_output = False

            # input data checks
            assert (
                x_categoricals is not None
            ), "x_categoricals must be provided."  # groups(fueltype)
            categorical_group_variables = [
                name for names in categorical_groups.values() for name in names
            ]
            if len(categorical_groups) > 0:
                assert all(
                    name in embedding_sizes for name in categorical_groups
                ), "categorical_groups must be in embedding_sizes."
                assert not any(
                    name in embedding_sizes for name in categorical_group_variables
                ), "group variable in categorical_group must not be in embedding_sizes"
                assert all(
                    name in x_categoricals for name in categorical_group_variables
                ), "group variables in categorical_groups must be in x_categoricals."
            assert all(
                name in embedding_sizes
                for name in embedding_sizes
                if name not in categorical_group_variables
            ), (
                "all variables in embedding_sizes must be in x_categoricals - but only if"
                "not already in categorical_groups."
            )
        else:
            assert (
                x_categoricals is None and len(categorical_groups) == 0
            ), " If embedding_sizes is not a dictionary, categorical_groups and x_categoricals must be empty."
            # number embeddings based on order
            embedding_sizes = {
                str(name): size for name, size in enumerate(embedding_sizes)
            }
            x_categoricals = list(embedding_sizes.keys())
            self.concat_output = True

        # infer embedding size if not determined
        self.embedding_sizes = {
            name: (size, get_embedding_size(size)) if isinstance(size, int) else size
            for name, size in embedding_sizes.items()
        }
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.x_categoricals = x_categoricals

        self.init_embeddings()

    def init_embeddings(self):
        self.embeddings = nn.ModuleDict()

        # print(f"{self.embeddings=}")

        for name in self.embedding_sizes.keys():
            #   print(f"{name=}")
            #  print(f"{self.embedding_sizes[name]=}")
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                #       print(f"{self.max_embedding_size=}")
                embedding_size = min(embedding_size, self.max_embedding_size)
            #      print(f"{embedding_size=}")
            # convert to list to become mutable
            #   print(self.embedding_sizes[name])
            self.embedding_sizes[name] = list(self.embedding_sizes[name])
            #   print(f"{self.embedding_sizes[name]=}")
            self.embedding_sizes[name][1] = embedding_size
            #   print(f"{self.embedding_sizes[name][1]=}")

            if name in self.categorical_groups:  # embedding bag if related embedding
                self.embeddings[name] = TimeDistributedEmbeddingBag(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    mode="sum",
                    batch_first=True,
                )
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    padding_idx=padding_idx,
                )

    def names(self):
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str):
        return self.embeddings[name]

    @property
    def input_size(self) -> Union[Dict[str, int], int]:
        if self.concat_output:
            return sum(s[1] for s in self.embedding_sizes.values())
        else:
            return {name: s[1] for name, s in self.embedding_sizes.items()}

    @property
    def output_size(self) -> Union[Dict[str, int], int]:
        if self.concat_output:
            return sum([s[1] for s in self.embedding_sizes.values()])
        else:
            return {name: s[1] for name, s in self.embedding_sizes.items()}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_vectors = {}

        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                input_vectors[name] = emb(
                    x[
                        ...,  # select all preceding dimensions
                        [
                            self.x_categoricals.index(cat_name)
                            for cat_name in self.categorical_groups[name]
                        ],
                    ]
                )
            else:
                input_vectors[name] = emb(x[..., self.x_categoricals.index(name)])
        if self.concat_output:
            return torch.cat(list(input_vectors.values()), dim=-1)
        else:
            return input_vectors
