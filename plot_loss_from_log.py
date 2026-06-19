#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re

import matplotlib.pyplot as plt


LOSS_RE = re.compile(
    r"Epoch\s+(\d+):\s+train\s+=\s+([+-]?\d+(?:\.\d+)?)"
    r".*?\|\s+dev\s+=\s+([+-]?\d+(?:\.\d+)?)"
)


def parse_losses(log_path):
    epochs = []
    train_losses = []
    dev_losses = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = LOSS_RE.search(line)
            if not match:
                continue
            epoch, train_loss, dev_loss = match.groups()
            epochs.append(int(epoch))
            train_losses.append(float(train_loss))
            dev_losses.append(float(dev_loss))

    return epochs, train_losses, dev_losses


def main():
    parser = argparse.ArgumentParser(
        description="Plot train/dev loss from trainer.log"
    )
    parser.add_argument("log_path")
    parser.add_argument(
        "--output",
        default="loss_vs_epoch.png",
        help="Output image path",
    )
    args = parser.parse_args()

    epochs, train_losses, dev_losses = parse_losses(args.log_path)
    if not epochs:
        raise RuntimeError("No epoch loss records found in log file")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train", color="red")
    plt.plot(epochs, dev_losses, label="Dev", color="blue")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved {args.output} with {len(epochs)} epochs")


if __name__ == "__main__":
    main()
