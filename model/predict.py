import json
import argparse

import torch
from transformers import RobertaTokenizer

from model import Seq2seq, Encoder, Decoder


argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input", required=True, help="Input file path")
argparser.add_argument(
    "-o", "--output", required=True, help="Output file path"
)


def load_tokenizer(path: str) -> RobertaTokenizer:
    return RobertaTokenizer.from_pretrained(path)


def predict(input, target):
    encoder = Encoder(
        source_tokenizer_meta["max_len"],
        source_tokenizer_meta["vocab_size"],
        128,
        128,
    )

    decoder = Decoder(
        target_tokenizer_meta["max_len"],
        target_tokenizer_meta["vocab_size"],
        128,
        128,
    )

    model = Seq2seq(encoder, decoder, device).to(device)
    model.load_state_dict(
        torch.load(
            "../model/model_saves/model_49_0.6721043531006212_lr_0.0001_batches_64.pt",
            map_location=device,
        ),
    )
    # model.eval()
    # with torch.no_grad():
    output = model(input, target, is_training=False)

    return output.argmax(dim=2).tolist()


if __name__ == "__main__":
    args = argparser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    source_tokenizer_path = "../data/source_tokenizer"
    target_tokenizer_path = "../data/target_tokenizer"

    source_tokenizer = load_tokenizer(f"{source_tokenizer_path}/tokenizer")
    source_tokenizer_meta = json.load(
        open(f"{source_tokenizer_path}/tokenizer_meta.json", "r")
    )
    target_tokenizer = load_tokenizer(f"{target_tokenizer_path}/tokenizer")
    target_tokenizer_meta = json.load(
        open(f"{target_tokenizer_path}/tokenizer_meta.json", "r")
    )

    with open(args.input, "r") as f:
        lines = f.readlines()
        lines = map(str.strip, lines)
        lines = [f"[STR]{x}[END]" for x in lines]

    tokenizer_params = {
        "add_special_tokens": False,
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }
    tokenized_source = source_tokenizer(
        lines, max_length=source_tokenizer_meta["max_len"], **tokenizer_params
    )["input_ids"].to(device)

    target = torch.ones(
        tokenized_source.size(0),
        target_tokenizer_meta["max_len"],
        device=device,
    ).to(torch.long)
    output = predict(tokenized_source, target)

    target_output = target_tokenizer.batch_decode(output)
    target_output = [x.replace("[END]", "") for x in target_output]
    with open(args.output, "w") as f:
        for iline, oline in zip(lines, target_output):
            f.write(f"{iline.replace('[STR]','').replace('[END]',' ')} {oline}\n")
