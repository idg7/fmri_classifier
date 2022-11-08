from torch import nn, Tensor
from .positional_encoding import PositionalEncoding


class TransformerDecoderClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_cls: int, nheads: int, rnn_num_layers: int) -> None:
        super().__init__()
        self.to_dim = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=True, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=rnn_num_layers)
        self.classifier = nn.Linear(hidden_dim, num_cls)

    def forward(self, x: Tensor) -> Tensor:
        # project the original dim to the decoder dim
        b, t, d = x.shape
        x = x.reshape(-1, d)
        h = self.to_dim(x)
        h = h.reshape(b, t, d)

        # Add positional encoding and send as input to the decoder
        h = self.positional_encoding(h)
        h = self.decoder(h, h)

        # Classify only the last TR embedding
        return self.classifier(h[:,-1])
