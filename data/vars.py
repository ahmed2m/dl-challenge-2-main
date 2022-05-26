import random

SEED = 3407
random.seed(SEED)
import datetime
import itertools


MIN_YEAR = 1800  # inclusive
MAX_YEAR = 2200  # inclusive
itom = {
    i: f"[{datetime.date(2008, i, 1).strftime('%B')[:3].upper()}]" for i in range(1, 13)
}
itod = {
    i: f"[{datetime.date(2022, 5, i).strftime('%A')[:3].upper()}]" for i in range(1, 8)
}
itodec = {
    i: f"[{datetime.date(i, 1, 1).strftime('%Y').upper()}]"
    for i in range(MIN_YEAR, MAX_YEAR + 10, 10)
}
all_possible_dates = itertools.product(
    range(1, 32), range(1, 13), range(MIN_YEAR, MAX_YEAR + 1)
)
