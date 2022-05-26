from vars import all_possible_dates
from util import get_all_tags, check_date_existence
from typing import Iterable
import pathlib

data_folder = pathlib.Path(__file__).parent.resolve()
data_path = f"{data_folder}/data.txt"


def main(data_path: str, dates: Iterable = all_possible_dates) -> None:
    with open(data_path, mode="w") as wfp:
        wfp.write("\n".join(get_all_tags(x) for x in dates if check_date_existence(x)))


if __name__ == "__main__":
    main(data_path=data_path)
