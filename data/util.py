from typing import Tuple
import datetime

TEXT_FORMAT = "{month} {decade} {leap} {day} {date}"


# Reference 1: https://www.cse.unsw.edu.au/~cs1511/17s2/week02/09_leapYear/


def is_leap_year(year: int):
    # Reference 1.
    return ((year % 100 != 0) and (year % 4 == 0)) or (year % 400 == 0)


def get_day_tag(date: Tuple[int, int, int]):
    day = datetime.date(*date[::-1]).strftime("%a").upper()
    return f"[{day}]"


def get_month_tag(date: Tuple[int, int, int]):
    month = datetime.date(*date[::-1]).strftime("%b").upper()
    return f"[{month}]"


def get_decade_tag(date: Tuple[int, int, int]):
    return f"[{date[-1] // 10}]"


def check_date_existence(date: Tuple[int, int, int]):
    try:
        datetime.datetime(*date[::-1])
        return True
    except ValueError:
        return False


def get_all_tags(date: Tuple[int, int, int]):
    global TEXT_FORMAT
    date_str = f"{date[0]}-{date[1]}-{date[2]}"
    result = TEXT_FORMAT.format(
        day=get_day_tag(date),
        month=get_month_tag(date),
        leap=f"[{str(is_leap_year(date[-1]))}]",
        decade=get_decade_tag(date),
        date=date_str,
    )
    return result

