import argparse

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import tqdm

import config
import optimization as optim
from model.dataset import MovieDataset, movie_collate_fn
from model.movie_classification import MovieClassification

DEVICE = None


# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--model-config-path", type=str)
parser.add_argument("--train-config-path", type=str)
parser.add_argument("--vocab-path", type=str)
parser.add_argument("--train-data-path", type=str)
parser.add_argument("--test-data-path", type=str)
# fmt: on


def train_epoch(
    epoch,
    model,
    criterion,
    optimizer,
    train_loader,
):
    model.train()
    losses = []

    with tqdm.tqdm(total=len(train_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(DEVICE), value)

            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)

            loss = criterion(outputs, labels)
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")

    return np.mean(losses)


def eval_epoch():
    pass


def main(args):
    model_config = config.Config.load(args.model_config_path)
    train_config = config.Config.load(args.train_config_path)
    vocab = spm.SentencePieceProcessor()
    vocab.load(args.vocab_path)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[+] Device:", DEVICE)

    train_dataset = MovieDataset(vocab, args.train_data_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=movie_collate_fn,
    )

    test_dataset = MovieDataset(vocab, args.test_data_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=movie_collate_fn,
    )

    print("[+] Num Train Dataset: ", len(train_dataset))
    print("[+] Num Train Data Loader: ", len(train_loader))
    print("[+] Num Test Dataset: ", len(test_dataset))
    print("[+] Num Test Data Loader: ", len(test_loader))

    model = MovieClassification(
        sequence_length=model_config.sequence_length,
        num_layers=model_config.num_layers,
        hidden_dim=model_config.hidden_dim,
        num_heads=model_config.num_heads,
        head_dim=model_config.head_dim,
        feed_forward_dim=model_config.feed_forward_dim,
        dropout_prob=model_config.dropout_prob,
        layer_norm_eps=model_config.layer_norm_eps,
        num_classes=model_config.num_classes,
    )

    criterion = torch.nn.CrossEntropyLoss()

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": train_config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(
        param_groups,
        lr=train_config.learning_rate,
        eps=train_config.adam_epsilon,
    )

    num_training_steps = len(train_loader) * train_config.epoch
    scheduler = optim.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    for epoch in tqdm.trange(train_config.epoch, desc="Epoch"):
        loss = train_epoch(
            epoch=epoch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
