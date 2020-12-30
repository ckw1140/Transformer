import json
from copy import deepcopy
from typing import Any, Dict, Type, TypeVar


class Config:
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
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feed_forward_dim = feed_forward_dim
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.num_classes = num_classes

    @classmethod
    def from_dict(cls, dict_object, **kwargs):
        dict_object.update(kwargs)
        return cls(**dict_object)

    @classmethod
    def from_json(cls, json_file_path, **kwargs):
        with open(json_file_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), **kwargs)
    
    def to_dict(self):
        return deepcopy(self.__dict__)
