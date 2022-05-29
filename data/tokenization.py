import os
import sys
import argparse
import json
import pickle

from transformers import RobertaTokenizer

from util import get_day_tag, get_month_tag

# https://stackoverflow.com/a/20959363/7523525
# to be able to run from the outside directory
# python model/train.py -args
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append(os.path.abspath("../."))


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    default="data.txt",
    help="Input file to tokenize",
)

args = parser.parse_args()


def gen_all_output_months():
    list = []
    for month in range(1, 13):
        list.append(f"-{month}-")
    return list


def gen_all_month_conditions() -> list[str]:
    months = []
    for i in range(1, 13):
        months.append(get_month_tag((1, i, 2022)))
    return months


def gen_all_day_conditions() -> list[str]:
    days = []
    for i in range(1, 8):
        days.append(get_day_tag((i, 1, 2022)))
    return days


def get_all_special_tokens(
    is_target: bool,
) -> tuple[dict[str, str], list[str]]:
    special = {
        "pad_token": "[PAD]",
        "bos_token": "[STR]",
        "eos_token": "[END]",
    }
    common = list(special.values()) + [str(x) for x in list(range(0, 10))]
    target = gen_all_output_months()  # -1-
    source = (
        gen_all_month_conditions()  # [JAN]
        + gen_all_day_conditions()  # [MON]
        + ["[True]", "[False]"]
        + ["[", "]"]
    )
    if is_target:
        return special, common + target

    return special, common + source


def parse_input(input: list[str]) -> tuple[list[str], list[str]]:
    map_input = map(str.strip, input)
    new_i, new_o = [], []
    for x in map_input:
        new_i.append(f"[STR]{x[:x.rfind(']')+1]}[END]")
        new_o.append(f"[STR]{x[x.rfind(']')+2:]}[END]")
    return new_i, new_o


def gen_tokens(path, is_target):
    # given the problem already known constraint about the data
    # char-by-char tokenizating is not really necessary
    # simple dictionary built from observations is sufficient
    _, special_tokens = get_all_special_tokens(is_target)
    vocab = dict(zip(special_tokens, range(len(special_tokens))))
    if not os.path.exists(path):
        os.mkdir(path)

    with open(f"{path}/vocab.json", "w") as f:
        json.dump(vocab, f, separators=(",", ":"))
    with open(f"{path}/merges.txt", "w") as f:
        f.write("# dummy file for roberta tokenizer\n")


def get_tokenized_data(input, path, is_target):
    special_dict, special_tokens = get_all_special_tokens(is_target)
    tokenizer_bpe = RobertaTokenizer.from_pretrained(
        path,
        do_lower_case=False,
        bos_token=special_dict["bos_token"],
        eos_token=special_dict["eos_token"],
        pad_token=special_dict["pad_token"],
        additional_special_tokens=special_tokens,
    )
    tokenized = tokenizer_bpe(
        input, add_special_tokens=False, padding=True, return_tensors="pt"
    )

    tokenizer_bpe.save_pretrained(f"{path}/tokenizer")
    return (
        tokenized["input_ids"],  # tokenized data
        len(special_tokens),  # vocab size
        tokenized["input_ids"].size(1),  # max length
    )


if __name__ == "__main__":
    lines = []
    with open(args.input) as f:
        lines = f.readlines()
    source, target = parse_input(lines)
    gen_tokens("source_tokenizer", is_target=False)
    gen_tokens("target_tokenizer", is_target=True)

    if not os.path.exists("source_tokenizer"):
        os.mkdir("source_tokenizer")
    if not os.path.exists("target_tokenizer"):
        os.mkdir("target_tokenizer")

    with open("source_tokenizer/tokenized_input.pkl", "wb") as dump:
        tokenized_source, src_vocab_size, src_max_len = get_tokenized_data(
            source,
            "source_tokenizer",
            is_target=False,
        )
        pickle.dump(
            tokenized_source,
            file=dump,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        with open("source_tokenizer/tokenizer_meta.json", "w") as f:
            json.dump(
                dict(
                    {
                        "vocab_size": src_vocab_size,
                        "max_len": src_max_len,
                    }
                ),
                f,
                indent=4,
            )

    with open("target_tokenizer/tokenized_output.pkl", "wb") as dump:
        tokenized_target, tgt_vocab_size, tgt_max_len = get_tokenized_data(
            target,
            "target_tokenizer",
            is_target=True,
        )
        pickle.dump(
            tokenized_target,
            file=dump,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        with open("target_tokenizer/tokenizer_meta.json", "w") as f:
            json.dump(
                dict(
                    {
                        "vocab_size": tgt_vocab_size,
                        "max_len": tgt_max_len,
                    }
                ),
                f,
                indent=4,
            )
