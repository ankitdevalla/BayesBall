from __future__ import annotations

import datetime as dt


def season_from_date(d: dt.date) -> str:
    # NBA season starts around October; use Oct 1 as cutoff.
    if d.month >= 10:
        start_year = d.year
    else:
        start_year = d.year - 1
    end_year = (start_year + 1) % 100
    return f"{start_year}-{end_year:02d}"
