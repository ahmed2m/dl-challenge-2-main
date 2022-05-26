# utility evaluation functions
import os
import sys
from typing import Tuple

sys.path.append(os.path.abspath('../.'))

from data.util import get_decade_tag, get_month_tag, is_leap_year, check_date_existence, get_day_tag

def parse_date(date: str) -> Tuple[Tuple[int, int, int], Tuple[str]]:
    """[WED] [JAN] [False] [180] 1-1-1800"""
    tokens = date.split()
    return (tuple(int(x) for x in tokens[-1].split('-')), tuple(tokens[:-1]))


def print_msg(verbose, msg):
    if verbose:
        print(msg)


def validate_date(date: Tuple[int, int, int], conditions: Tuple[str, str, str, str], verbose=True) -> bool:
    if not check_date_existence(date):
        return False

    day, mon, leap, decade = conditions

    if eval(leap[1:-1]):
        if not is_leap_year(date[-1]):
            print_msg(verbose=verbose, msg=f'{date[-1]} is not a leap year while it should be.')
            return False
    else:
        if is_leap_year(date[-1]):
            print_msg(verbose=verbose, msg=f'{date[-1]} is not a leap year while it should not be.')
            return False
    
    valid_month = get_month_tag(date)
    if mon != valid_month:
        print_msg(verbose=verbose, msg=f'{mon[1:-1]} is not True, it should be {valid_month}.')
        return False

    valid_decade = get_decade_tag(date)
    if decade != valid_decade:
        print_msg(verbose=verbose, msg=f'{decade[1:-1]} is not True, it should be {valid_decade}.')
        return False

    valid_day = get_day_tag(date)
    if day != valid_day:
        print_msg(verbose=verbose, msg=f'{day[1:-1]} is not True, it should be {valid_day}.')
        return False
    
    return True

def parse_and_validate_date(date: str):
    return validate_date(*parse_date(date))

def validate_dates_from_file(path: str):
    valid_count, invalid_count = 0, 0
    with open(os.path.abspath(path)) as rfp:
        for line in rfp.readlines():
            is_valid = parse_and_validate_date(line.strip())
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
    return {'valid_counts': valid_count, 'invalid_counts': invalid_count}


# if __name__ == '__main__':
#     validate_dates_from_file('../data/data.txt')