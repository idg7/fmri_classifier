from torch import nn, Tensor, randn, cat


class TransformerClassifier(nn.Module):
    def __init__(self, dim: int,
                 encoder: nn.TransformerEncoder,
                 positional_encoding: nn.Module,
                 classifier: nn.Module) -> None:
        super(TransformerClassifier).__init__()
        self.positional_encoding = positional_encoding
        self.cls_token = nn.Parameter(randn(1, 1, dim))
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape

        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = cat((cls_tokens, x), dim=1)
        positioned_x = self.positional_encoding(x)

        embeddings = self.encoder(positioned_x)

        cls = embeddings[:, 0, :]
        cls = cls.squeeze(1)
        return cls
