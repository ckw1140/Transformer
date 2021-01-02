import torch
import torch.nn as nn

from model.transformer import Transformer


class MovieClassification(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_layers,
        hidden_dim,
        num_heads,
        head_dim,
        feed_forward_dim,
        dropout_prob,
        layer_norm_eps,
        num_classes,
    ):
        super(MovieClassification, self).__init__()

        self.transformer = Transformer(
            sequence_length=sequence_length,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_prob=dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.linear = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(
        self,
        enc_inputs,
        dec_inputs,
    ):
        # outputs: [batch_size, sequence_length, hidden_dim]
        outputs, _, _, _ = self.transformer(enc_inputs, dec_inputs)

        # outputs: [batch_size, hidden_dim]
        outputs, _ = torch.max(outputs, dim=1)

        # outputs: [batch_size, num_classes]
        outputs = self.linear(outputs)

        return outputs
    
    def save(self, epoch, loss, score, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "score": score,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"], save["score"]
