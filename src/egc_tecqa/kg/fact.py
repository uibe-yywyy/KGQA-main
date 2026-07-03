from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable


def parse_date(value: str | int | date | None) -> date | None:
    """Parse year/month/day-like values into dates.

    Month and year inputs are normalized to the first day of the period. This
    is enough for prototype ranking; interval-aware checks live in executor.
    """

    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value!r}")


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str
    relation: str
    object: str
    start_time: date | None = None
    end_time: date | None = None
    source: str | None = None

    @classmethod
    def from_values(
        cls,
        fact_id: str,
        subject: str,
        relation: str,
        object: str,
        start_time: str | int | date | None = None,
        end_time: str | int | date | None = None,
        source: str | None = None,
    ) -> "Fact":
        start = parse_date(start_time)
        end = parse_date(end_time) if end_time is not None else start
        return cls(fact_id, subject, relation, object, start, end, source)

    @property
    def entities(self) -> set[str]:
        return {self.subject, self.object}

    @property
    def representative_time(self) -> date | None:
        if self.start_time is None:
            return self.end_time
        if self.end_time is None or self.end_time == self.start_time:
            return self.start_time
        ordinal_mid = (self.start_time.toordinal() + self.end_time.toordinal()) // 2
        return date.fromordinal(ordinal_mid)

    def shares_entity_with(self, other: "Fact") -> bool:
        return bool(self.entities & other.entities)

    def as_tuple(self) -> tuple[str, str, str, str | None, str | None]:
        return (
            self.subject,
            self.relation,
            self.object,
            self.start_time.isoformat() if self.start_time else None,
            self.end_time.isoformat() if self.end_time else None,
        )

    def text(self) -> str:
        times = []
        if self.start_time:
            times.append(self.start_time.isoformat())
        if self.end_time and self.end_time != self.start_time:
            times.append(self.end_time.isoformat())
        time_text = " to ".join(times)
        return f"{self.subject} {self.relation} {self.object} {time_text}".strip()


def normalize_facts(raw_facts: Iterable[tuple]) -> list[Fact]:
    facts: list[Fact] = []
    for idx, row in enumerate(raw_facts):
        if len(row) == 4:
            subject, relation, object, timestamp = row
            facts.append(Fact.from_values(str(idx), subject, relation, object, timestamp))
        elif len(row) == 5:
            subject, relation, object, start, end = row
            facts.append(Fact.from_values(str(idx), subject, relation, object, start, end))
        else:
            raise ValueError(f"Expected 4 or 5 columns, got {len(row)}: {row!r}")
    return facts

