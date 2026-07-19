"""Shared initialization-calendar and W34 lead-window date utilities.

This module is the single source of truth for matched HeatCast/ECMWF ENS
Monday/Thursday initializations and week-3, week-4, and W34 date arithmetic.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Sequence, Tuple


MJJAS_MONTHS: Tuple[int, ...] = (5, 6, 7, 8, 9)
WEEK3_LEADS: Tuple[int, ...] = tuple(range(15, 22))
WEEK4_LEADS: Tuple[int, ...] = tuple(range(22, 29))
W34_LEADS: Tuple[int, ...] = tuple(range(15, 29))


def valid_dates(initialization: date, leads: Sequence[int] = W34_LEADS) -> Tuple[date, ...]:
    """Return valid UTC dates for integer day leads from one initialization."""
    normalized = tuple(int(lead) for lead in leads)
    if not normalized:
        raise ValueError("Lead window cannot be empty.")
    if any(lead < 0 for lead in normalized):
        raise ValueError(f"Lead days must be non-negative, got {normalized}.")
    return tuple(initialization + timedelta(days=lead) for lead in normalized)


def window_falls_in_months(
    initialization: date,
    leads: Sequence[int] = W34_LEADS,
    months: Sequence[int] = MJJAS_MONTHS,
) -> bool:
    """Return whether every valid date in a lead window is in ``months``."""
    allowed = {int(month) for month in months}
    if not allowed or any(month < 1 or month > 12 for month in allowed):
        raise ValueError(f"Months must be calendar month numbers, got {sorted(allowed)}.")
    return all(value.month in allowed for value in valid_dates(initialization, leads))


def mjjas_mon_thu(
    year: int,
    *,
    require_full_w34: bool = True,
) -> Iterable[date]:
    """Yield MJJAS Monday/Thursday initializations for matched ENS scoring.

    By default, dates are restricted so that every lead from 15 through 28
    remains in MJJAS. ``require_full_w34=False`` exposes the legacy unfiltered
    MJJAS initialization calendar for diagnostics only.
    """
    day = date(int(year), MJJAS_MONTHS[0], 1)
    end = date(int(year), MJJAS_MONTHS[-1], 30)
    while day <= end:
        if day.weekday() in (0, 3) and (
            not require_full_w34 or window_falls_in_months(day, W34_LEADS, MJJAS_MONTHS)
        ):
            yield day
        day += timedelta(days=1)
