import torch

from model.transformer import Encoder


def test_encoder():
    batch_size = 64
    sequence_length=16
    num_layers=12
    hidden_dim=8
    num_heads=4
    head_dim=2
    feed_forward_dim = 32
    dropout_prob = 0.3
    layer_norm_eps = 1e-4

    encoder = Encoder(
        sequence_length=sequence_length,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feed_forward_dim=feed_forward_dim,
        dropout_prob=dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )

    inputs = torch.arange(
        batch_size * sequence_length,
        device="cpu",
        dtype=torch.long,
    )
    inputs = inputs.view(batch_size, sequence_length)
    outputs, attention_probs = encoder(inputs)

    assert outputs.size() == (batch_size, sequence_length, hidden_dim)
    assert attention_probs[0].size() == (batch_size, num_heads, sequence_length, sequence_length)
