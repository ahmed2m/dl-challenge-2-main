import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from model import Seq2seq, Encoder, Decoder

# https://stackoverflow.com/a/20959363/7523525
# to be able to run from the outside directory
# python model/train.py -args
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append(os.path.abspath("../."))

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source",
    default="../data/source_tokenizer",
    help="Input file to tokenize",
)
parser.add_argument(
    "-t",
    "--target",
    default="../data/target_tokenizer",
    help="Output file to save tokenized input data",
)
args = parser.parse_args()


def train_model(
    model, input, target, num_epochs, learning_rate, batch_size, device
):
    loss_ = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_loss = 10e99
    last_epoch = 0
    loss_log = []

    train_loader = DataLoader(
        dataset=list(zip(input, target)),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    steps_loss = []
    tokenizer = RobertaTokenizer.from_pretrained("../data/target_tokenizer")
    for epoch in tqdm(range(num_epochs)):
        sum_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data, target)
            output_dim = outputs.shape[-1]

            # to ignore the first token (start of sequence) in loss calculation
            target_r = target[:, 1:].reshape(-1)
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            loss = loss_(outputs, target_r)
            loss.backward()

            optimizer.step()
            sum_loss += loss.item()
            steps_loss.append(loss.item())

        print(
            tokenizer.decode(outputs.argmax(2).tolist()[0]),
            "\n",
            tokenizer.decode(target.tolist()[0]),
        )

        current_loss = sum_loss / len(train_loader)
        print("Epoch: %d, loss: %1.5f" % (epoch, current_loss))

        model_meta = f"_lr_{learning_rate}_batches_{batch_size}"
        if current_loss < last_loss:
            # print("Saving...")
            if last_epoch != 0:
                os.remove(
                    f"model_saves/model_{last_epoch}_{last_loss}{model_meta}.pt"
                )
            torch.save(
                model.state_dict(),
                f"model_saves/model_{epoch}_{current_loss}{model_meta}.pt",
            )
            last_loss = current_loss
            last_epoch = epoch

        loss_log.append(current_loss)

    plt.figure(1)
    plt.plot(steps_loss)
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    plt.savefig(
        f"stepslossplot_lr-{learning_rate}_batch-{batch_size}_{num_epochs}.png"
    )
    plt.show()
    with open("steps_loss.txt", "w") as f:
        for loss in steps_loss:
            f.write(f"{loss}\n")
    return model, loss_log


def get_tokenizer_metadata(path):
    return json.load(open(f"{path}/tokenizer_meta.json", "r"))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f"Using device: {device}")

    source = pickle.load(open(f"{args.source}/tokenized_input.pkl", "rb"))
    target = pickle.load(open(f"{args.target}/tokenized_output.pkl", "rb"))
    src_tokenizer = get_tokenizer_metadata(args.source)
    tgt_tokenizer = get_tokenizer_metadata(args.target)
    source = source
    target = target

    encoder = Encoder(
        src_tokenizer["max_len"],
        src_tokenizer["vocab_size"],
        128,
        128,
    )

    decoder = Decoder(
        tgt_tokenizer["max_len"],
        tgt_tokenizer["vocab_size"],
        128,
        128,
    )

    model = Seq2seq(encoder, decoder, device).to(device)

    learning_rate = 0.0001
    num_epochs = 50
    batch_size = 64
    model, loss_log = train_model(
        model,
        source,
        target,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
    )

    plt.figure(2)
    plt.plot(loss_log)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig(
        f"lossplot_lr-{learning_rate}_batch-{batch_size}_{num_epochs}.png"
    )
    plt.show()
    with open("loss_log.txt", "w") as f:
        for loss in loss_log:
            f.write(f"{loss}\n")
# command line arguments
# python model/train.py -s ../data/source_tokenizer -t ../data/target_tokenizer
