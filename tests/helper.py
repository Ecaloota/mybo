from datetime import datetime, timedelta
from typing import Optional


def generate_datetime_range(
    start_datetime: datetime, end_datetime: datetime, delta: Optional[timedelta] = timedelta(minutes=5)
) -> list[datetime]:
    """Generates a list of datetimes spanning start to end"""

    date_list = []
    current_date = start_datetime

    while current_date <= end_datetime:
        date_list.append(current_date)
        current_date += delta

    return date_list


def fuzzy_equal(a, b, tol=1e-6) -> bool:
    """Returns true if abs(a-b) < tol, False otherwise"""

    return abs(a - b) < tol
