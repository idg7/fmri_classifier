from torch import nn, Tensor, LongTensor, stack
from typing import List


class IndividualizedModel(nn.Module):
    """
    A model with individual input layers per subject
    """
    def __init__(self, input_layers_dims: List[int], inner_mode_input_dim: int, inner: nn.Module) -> None:
        super().__init__()
        self.input_layers_dims = input_layers_dims
        self.inner_mode_input_dim = inner_mode_input_dim
        self.inner = inner
        self.input_layers = []
        
        # Create a list of input layers per individual subject
        for input_dim in self.input_layers_dims:
            self.input_layers.append(nn.Linear(input_dim, inner_mode_input_dim))
        
        self.input_layers = nn.ModuleList(self.input_layers)
        print(len(self.input_layers))

    def forward(self, x: List[Tensor], subj_id: LongTensor) -> Tensor:
        # project X from the individual subject dimension to a shared size dim
        projected_x = [None] * len(x)
        for i in range(len(x)):
            projected_x[i] = self.input_layers[subj_id[i]](x[i].float())
        projected_x = stack(projected_x)
        return self.inner(projected_x)