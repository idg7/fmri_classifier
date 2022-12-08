from torch import nn, Tensor
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_cls: int, num_layers: int) -> None:
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append((f'affine{i}', nn.Linear(input_dim, hidden_dim)))
            layers.append((f'relu{i}',nn.ReLU()))
        
        self.layers = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(hidden_dim, num_cls)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.squeeze(1)
        x = self.layers(x)
        return self.classifier(x)
