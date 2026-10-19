"""How rows are spread across partitions (:class:`HashPartitioning`), and how an
operator needs them spread (:class:`ClusteredDistribution`)."""

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "ClusteredDistribution",
    "HashPartitioning",
]


@dataclass(frozen=True)
class HashPartitioning:
    """Rows are hash-partitioned on ``columns`` into ``num_partitions`` partitions.

    Rows with equal values in ``columns`` are in the same partition. Nothing is
    implied about row order or how many blocks a partition spans.
    """

    columns: Tuple[str, ...]
    num_partitions: int

    def __post_init__(self):
        assert self.columns, "HashPartitioning needs at least one column"
        assert self.num_partitions > 0, self.num_partitions


@dataclass(frozen=True)
class ClusteredDistribution:
    """Rows with equal values in ``columns`` must be in the same partition."""

    columns: Tuple[str, ...]

    def __post_init__(self):
        assert self.columns, "ClusteredDistribution needs at least one column"

    def is_satisfied_by(self, partitioning: Optional[HashPartitioning]) -> bool:
        """Whether ``partitioning`` already clusters rows as required.

        Partitioning on a subset of ``columns`` suffices: rows equal on all of
        ``columns`` are also equal on the subset, so they share a partition.
        """
        if partitioning is None:
            return False
        return set(partitioning.columns).issubset(self.columns)
