"""Tests for tracking output partitioning."""

from typing import Callable, Optional, Tuple

import numpy as np
import pyarrow as pa
import pytest

import ray
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import (
    ClusteredDistribution,
    HashPartitioning,
)
from ray.data._internal.logical.operators import Project
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data.aggregate import Count, Sum
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.dataset import Dataset
from ray.data.expressions import col
from ray.data.tests.conftest import *  # noqa: F401, F403
from ray.tests.conftest import *  # noqa: F401, F403

NUM_PARTITIONS = 4
NUM_ROWS = 120
# Reads both join sides, so PredicatePushdown keeps it above the join.
TWO_SIDED_PREDICATE = (col("lval") + col("rval")) > 100


@pytest.fixture(autouse=True)
def shuffle_v2(restore_data_context: DataContext) -> DataContext:
    restore_data_context.shuffle_strategy = ShuffleStrategy.SHUFFLE_V2
    return restore_data_context


def _from_columns(**columns) -> Dataset:
    """Six Arrow blocks with a schema known at plan time, like a Parquet read."""
    table = pa.table(columns)
    chunks = np.array_split(np.arange(table.num_rows), 6)
    return ray.data.from_arrow([table.take(indices) for indices in chunks])


def _left() -> Dataset:
    ids = np.arange(NUM_ROWS)
    return _from_columns(id=ids, grp=ids % 4, lval=ids * 2)


def _right() -> Dataset:
    # Two rows per key, so every group has more than one row.
    ids = np.arange(2 * NUM_ROWS)
    return _from_columns(rid=ids % NUM_ROWS, rval=ids * 3)


def _join(join_type: str = "inner", right: Optional[Dataset] = None) -> Dataset:
    return _left().join(
        _right() if right is None else right,
        join_type,
        on=("id",),
        right_on=("rid",),
        num_partitions=NUM_PARTITIONS,
    )


def _physical_dag(ds: Dataset) -> PhysicalOperator:
    return get_execution_plan(ds._logical_plan)[0].dag


def _has_aggregate_shuffle(dag: PhysicalOperator) -> bool:
    return any("HashAggregateMap" in op.name for op in dag.post_order_iter())


# ---------------------------------------------------------------------------
# Partitioning properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "partitioning_columns,required_columns,satisfied",
    [
        (("a",), ("a",), True),
        # A subset of the required columns suffices.
        (("a",), ("b", "a"), True),
        (("b", "a"), ("a", "b", "c"), True),
        (("a", "b"), ("a",), False),
        (("a",), ("b",), False),
        (None, ("a",), False),
    ],
)
def test_clustered_distribution_satisfaction(
    partitioning_columns: Optional[Tuple[str, ...]],
    required_columns: Tuple[str, ...],
    satisfied: bool,
):
    partitioning = None
    if partitioning_columns is not None:
        partitioning = HashPartitioning(partitioning_columns, 8)
    distribution = ClusteredDistribution(required_columns)
    assert distribution.is_satisfied_by(partitioning) == satisfied


@pytest.mark.parametrize(
    "join_type,key_column",
    [
        ("inner", "id"),
        ("left_outer", "id"),
        ("full_outer", "id"),
        ("left_semi", "id"),
        ("left_anti", "id"),
        # Right-preserving joins keep the right side's key column.
        ("right_outer", "rid"),
        ("right_semi", "rid"),
        ("right_anti", "rid"),
    ],
)
def test_join_output_partitioning(
    ray_start_regular_shared_2_cpus, join_type: str, key_column: str
):
    ds = _join(join_type)
    assert _physical_dag(ds).output_partitioning == HashPartitioning(
        (key_column,), NUM_PARTITIONS
    )
    assert key_column in ds.schema().names


def test_right_outer_join_with_key_named_left_column(ray_start_regular_shared_2_cpus):
    """The keys coalesce into ``rid_right``, and ``rid`` holds left data."""
    ids = np.arange(NUM_ROWS)
    left = _from_columns(id=ids, rid=pa.array([None] * NUM_ROWS, pa.int64()))
    joined = left.join(
        _right(),
        "right_outer",
        on=("id",),
        right_on=("rid",),
        num_partitions=NUM_PARTITIONS,
    )
    assert "rid_right" in joined.schema().names
    assert _physical_dag(joined).output_partitioning is None

    # ``rid`` is null on every row, so skipping the shuffle would split its group.
    ds = joined.groupby("rid").aggregate(Count(alias_name="c"))
    assert _has_aggregate_shuffle(_physical_dag(ds))
    assert ds.take_all() == [{"rid": None, "c": 2 * NUM_ROWS}]


def test_aggregate_partitioning(ray_start_regular_shared_2_cpus):
    ds = _left().groupby(["grp", "id"]).aggregate(Sum(on="lval", alias_name="s"))
    assert ds._logical_plan.dag.required_input_distribution == (
        ClusteredDistribution(("grp", "id"))
    )
    assert _physical_dag(ds).output_partitioning == HashPartitioning(
        ("grp", "id"), DataContext.get_current().default_hash_shuffle_parallelism
    )

    # A global aggregation has no groups to cluster.
    ds = _left().groupby(None).aggregate(Sum(on="lval", alias_name="s"))
    assert ds._logical_plan.dag.required_input_distribution is None
    assert _physical_dag(ds).output_partitioning is None


@pytest.mark.parametrize(
    "transform,keeps_partitioning",
    [
        pytest.param(lambda ds: ds.select_columns(["id", "rval"]), True, id="select"),
        pytest.param(
            lambda ds: ds.with_column("x", col("rval") + 1), True, id="add_column"
        ),
        pytest.param(
            lambda ds: ds.with_columns({"key": col("id"), "y": col("lval")}),
            True,
            id="copy_key",
        ),
        pytest.param(lambda ds: ds.filter(expr=TWO_SIDED_PREDICATE), True, id="filter"),
        # Fused into one map before being fused into the join's reduce.
        pytest.param(
            lambda ds: ds.filter(expr=TWO_SIDED_PREDICATE).with_column(
                "x", col("rval") * 2
            ),
            True,
            id="filter_then_add_column",
        ),
        pytest.param(
            lambda ds: ds.rename_columns({"id": "key"}), False, id="rename_key"
        ),
        pytest.param(
            lambda ds: ds.with_column("id", col("id") + 1), False, id="overwrite_key"
        ),
        pytest.param(
            lambda ds: ds.with_columns({"id": col("lval")}),
            False,
            id="other_column_as_key",
        ),
        pytest.param(lambda ds: ds.select_columns(["rval"]), False, id="drop_key"),
        pytest.param(
            lambda ds: ds.map_batches(lambda batch: batch), False, id="map_batches"
        ),
    ],
)
def test_map_output_partitioning(
    ray_start_regular_shared_2_cpus,
    transform: Callable[[Dataset], Dataset],
    keeps_partitioning: bool,
):
    partitioning = _physical_dag(transform(_join())).output_partitioning
    if keeps_partitioning:
        assert partitioning == HashPartitioning(("id",), NUM_PARTITIONS)
    else:
        assert partitioning is None


def test_project_preserves_columns_with_unknown_schema(
    ray_start_regular_shared_2_cpus,
):
    """Without an input schema, ``star()`` stays unexpanded and forwards columns."""
    opaque = _left().map_batches(lambda batch: batch)
    assert opaque.schema(fetch_if_missing=False) is None

    project = opaque.with_column("x", col("lval") + 1)._logical_plan.dag
    assert isinstance(project, Project) and project.has_star_expr()
    assert project.preserves_columns(["id", "grp"])

    project = opaque.with_column("id", col("id") + 1)._logical_plan.dag
    assert not project.preserves_columns(["id"])

    project = opaque.rename_columns({"id": "key"})._logical_plan.dag
    assert not project.preserves_columns(["id"])
    assert project.preserves_columns(["grp"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
