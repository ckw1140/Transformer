import json
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, path):
        self.vocab = vocab
        self.labels = []
        self.sentences = []

        with open(path, "r") as f:
            line_cnt = 0
            for line in f:
                line_cnt += 1

        with open(path, "r") as f:
            for i, line in enumerate(tqdm.tqdm(f, total=line_cnt, desc=f"Loading {path}", unit=" lines")):
                data = json.loads(line)
                self.labels.append(data["label"])
                self.sentences.append([vocab.piece_to_id(p) for p in data["doc"]])

    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(self.labels[index]),
            torch.tensor(self.sentences[index]),
            torch.tensor([self.vocab.piece_to_id("[BOS]")]),
        )


def movie_collate_fn(inputs):
    labels, enc_inputs, dec_inputs = list(zip(*inputs))

    enc_inputs = torch.nn.utils.rnn.pad_sequence(
        enc_inputs,
        batch_first=True,
        padding_value=0,
    )
    dec_inputs = torch.nn.utils.rnn.pad_sequence(
        dec_inputs,
        batch_first=True,
        padding_value=0,
    )

    return [
        torch.stack(labels, dim=0),
        enc_inputs,
        dec_inputs,
    ]
