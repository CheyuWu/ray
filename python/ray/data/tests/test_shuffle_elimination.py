"""Tests for tracking output partitioning and skipping redundant group-by shuffles."""

import collections
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import ray
from ray.data._internal.execution.interfaces import PhysicalOperator, TaskContext
from ray.data._internal.execution.operators.hash_aggregate import (
    HashAggregateOperator,
)
from ray.data._internal.execution.operators.hash_aggregate_v2 import (
    _make_complete_aggregating_block_fn,
)
from ray.data._internal.logical.interfaces import (
    ClusteredDistribution,
    HashPartitioning,
)
from ray.data._internal.logical.operators import Project
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data.aggregate import AbsMax, Count, Max, Sum
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.dataset import Dataset
from ray.data.expressions import col
from ray.data.grouped_data import GroupedData
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


def _local_aggregates(dag: PhysicalOperator) -> List[PhysicalOperator]:
    return [op for op in dag.post_order_iter() if "mode=complete" in op.name]


def _has_aggregate_shuffle(dag: PhysicalOperator) -> bool:
    return any("HashAggregateMap" in op.name for op in dag.post_order_iter())


def _pandas_groupby(
    ds: Dataset, keys: List[str], **aggs: Tuple[str, str]
) -> pd.DataFrame:
    return ds.to_pandas().groupby(keys, dropna=False).agg(**aggs).reset_index()


def _assert_same_rows(ds: Dataset, expected: pd.DataFrame, keys: List[str]) -> None:
    actual = ds.to_pandas()
    assert sorted(actual.columns) == sorted(expected.columns)
    actual = actual[expected.columns].sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


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


# ---------------------------------------------------------------------------
# Shuffle elimination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_disk", [False, True])
def test_groupby_after_join_skips_shuffle(
    ray_start_regular_shared_2_cpus, shuffle_v2: DataContext, use_disk: bool
):
    """TPC-H q3 shape: join, derive a column, group by a superset of the key."""
    shuffle_v2.use_disk_based_hash_shuffle = use_disk
    before_agg = _join().with_column("rev", col("lval") * col("rval"))
    ds = before_agg.groupby(["id", "grp"]).aggregate(
        Sum(on="rev", alias_name="rev"), Count(alias_name="c")
    )

    dag = _physical_dag(ds)
    assert dag.name == "HashAggregate(key_columns=('id', 'grp'), mode=complete)"
    (join_reduce,) = dag.input_dependencies
    assert join_reduce.name.endswith(
        f"JoinShuffleReduce(num_partitions={NUM_PARTITIONS})->Project"
    )
    assert join_reduce._emit_whole_partitions

    expected = _pandas_groupby(
        before_agg, ["id", "grp"], rev=("rev", "sum"), c=("rval", "size")
    )
    _assert_same_rows(ds, expected, ["id", "grp"])


@pytest.mark.parametrize("use_disk", [False, True])
def test_partition_spread_over_many_blocks(
    ray_start_regular_shared_2_cpus, shuffle_v2: DataContext, use_disk: bool
):
    """Each aggregation task gets a whole partition, however many blocks it spans."""
    shuffle_v2.use_disk_based_hash_shuffle = use_disk
    shuffle_v2.target_max_block_size = 256
    joined = _join()
    assert joined.materialize().num_blocks() > 4 * NUM_PARTITIONS

    # Reads every column, so no pruning Project is fused into the join's reduce
    # to merge its blocks back together.
    ds = joined.groupby(["id", "grp"]).aggregate(
        Sum(on="rval", alias_name="s"),
        Max(on="lval", alias_name="m"),
        Count(alias_name="c"),
    )
    (join_reduce,) = _physical_dag(ds).input_dependencies
    assert join_reduce.name.endswith(
        f"JoinShuffleReduce(num_partitions={NUM_PARTITIONS})"
    )

    # One aggregation task per (non-empty) join partition.
    ds = ds.materialize()
    (aggregate_stats,) = [
        stats
        for stats in ds._get_stats_summary().operators_stats
        if "mode=complete" in stats.operator_name
    ]
    assert aggregate_stats.task_rows.count == NUM_PARTITIONS

    expected = _pandas_groupby(
        joined, ["id", "grp"], s=("rval", "sum"), m=("lval", "max"), c=("rval", "size")
    )
    _assert_same_rows(ds, expected, ["id", "grp"])


@pytest.mark.parametrize(
    "transform,skips_shuffle",
    [
        pytest.param(
            lambda ds: ds.groupby("id", num_partitions=NUM_PARTITIONS),
            True,
            id="same_num_partitions",
        ),
        pytest.param(
            lambda ds: ds.filter(expr=TWO_SIDED_PREDICATE).groupby("id"),
            True,
            id="fusable_filter",
        ),
        pytest.param(lambda ds: ds.groupby("grp"), False, id="other_key"),
        pytest.param(
            lambda ds: ds.groupby("id", num_partitions=NUM_PARTITIONS + 1),
            False,
            id="different_num_partitions",
        ),
        pytest.param(
            lambda ds: ds.with_column("id", col("id") % 3).groupby("id"),
            False,
            id="key_overwritten",
        ),
        # A map not fused into the join's reduce passes on blocks one at a time.
        pytest.param(
            lambda ds: ds.filter(expr=TWO_SIDED_PREDICATE, concurrency=1).groupby("id"),
            False,
            id="unfusable_filter",
        ),
    ],
)
def test_aggregate_shuffle_elimination(
    ray_start_regular_shared_2_cpus,
    transform: Callable[[Dataset], GroupedData],
    skips_shuffle: bool,
):
    grouped = transform(_join())
    ds = grouped.aggregate(Sum(on="rval", alias_name="s"))
    assert _has_aggregate_shuffle(_physical_dag(ds)) == (not skips_shuffle)

    keys = [grouped._key]
    expected = _pandas_groupby(grouped._dataset, keys, s=("rval", "sum"))
    _assert_same_rows(ds, expected, keys)


def test_groupby_with_limit_after_sort_skips_shuffle(ray_start_regular_shared_2_cpus):
    """TPC-H q18 shape: the limit is behind the sort's shuffle, so it doesn't
    keep the projection after the join from fusing into the join's reduce."""
    joined = _join().select_columns(["id", "rval"])
    ds = (
        joined.groupby("id")
        .aggregate(Sum(on="rval", alias_name="s"))
        .sort("s", descending=True)
        .limit(5)
    )
    assert _local_aggregates(_physical_dag(ds))

    expected = _pandas_groupby(joined, ["id"], s=("rval", "sum")).nlargest(5, "s")
    _assert_same_rows(ds, expected, ["id"])


def test_left_outer_join_then_count(ray_start_regular_shared_2_cpus):
    """TPC-H q13 shape: unmatched keys must count as 0."""
    joined = _join("left_outer", right=_right().filter(expr=col("rid") < 60))
    ds = joined.groupby("id").aggregate(
        Count(on="rval", ignore_nulls=True, alias_name="c")
    )
    assert _local_aggregates(_physical_dag(ds))

    expected = _pandas_groupby(joined, ["id"], c=("rval", "count"))
    assert (expected["c"] == 0).sum() == 60
    _assert_same_rows(ds, expected, ["id"])


def test_right_outer_join_groups_by_right_key(ray_start_regular_shared_2_cpus):
    joined = _join("right_outer", right=_right().filter(expr=col("rid") >= 60))
    ds = joined.groupby("rid").aggregate(Sum(on="rval", alias_name="s"))
    assert _local_aggregates(_physical_dag(ds))
    _assert_same_rows(ds, _pandas_groupby(joined, ["rid"], s=("rval", "sum")), ["rid"])


def test_full_outer_join_null_keys_form_one_group(ray_start_regular_shared_2_cpus):
    """Null keys from either side coalesce into one key column, and must still
    land in one partition (and so one group)."""
    left = _from_columns(
        id=pa.array([1, 2, 3, None, None], pa.int64()), lval=np.arange(5)
    )
    right = _from_columns(rid=pa.array([2, 3, 4, None], pa.int64()), rval=np.arange(4))
    joined = left.join(
        right,
        "full_outer",
        on=("id",),
        right_on=("rid",),
        num_partitions=NUM_PARTITIONS,
    )
    ds = joined.groupby("id").aggregate(Count(alias_name="c"))
    assert _local_aggregates(_physical_dag(ds))

    expected = collections.Counter(row["id"] for row in joined.take_all())
    # Two null keys from the left, one from the right.
    assert expected[None] == 3
    assert {row["id"]: row["c"] for row in ds.take_all()} == expected


def test_second_groupby_after_skipped_shuffle_keeps_its_shuffle(
    ray_start_regular_shared_2_cpus,
):
    """A one-phase aggregation is a map, so it doesn't hand over whole
    partitions to a group-by after it."""
    joined = _join()
    first = joined.groupby("id").aggregate(Sum(on="rval", alias_name="s"))
    ds = first.groupby(["id", "s"]).aggregate(Count(alias_name="c"))
    dag = _physical_dag(ds)
    assert len(_local_aggregates(dag)) == 1
    assert _has_aggregate_shuffle(dag)

    expected = _pandas_groupby(joined, ["id"], s=("rval", "sum"))
    expected["c"] = 1
    _assert_same_rows(ds, expected, ["id"])


def test_groupby_after_groupby_skips_second_shuffle(ray_start_regular_shared_2_cpus):
    first = _left().groupby("grp").aggregate(Sum(on="lval", alias_name="s"))
    ds = first.groupby(["grp", "s"]).aggregate(Count(alias_name="c"))
    dag = _physical_dag(ds)
    assert dag.name == "HashAggregate(key_columns=('grp', 's'), mode=complete)"
    (first_reduce,) = dag.input_dependencies
    assert "HashAggregateReduce" in first_reduce.name

    expected = _pandas_groupby(_left(), ["grp"], s=("lval", "sum"))
    expected["c"] = 1
    _assert_same_rows(ds, expected, ["grp"])


def test_map_after_aggregation_runs_in_its_tasks(ray_start_regular_shared_2_cpus):
    # Op names are chained with ``->``, which a key column's name may contain.
    keys = ["id", "g->rp"]
    joined = _join().rename_columns({"grp": "g->rp"})
    ds = (
        joined.groupby(keys)
        .aggregate(Sum(on="rval", alias_name="s"))
        .with_column("s2", col("s") * 2)
    )
    assert _physical_dag(ds).name == (
        "HashAggregate(key_columns=('id', 'g->rp'), mode=complete)->Project"
    )

    expected = _pandas_groupby(joined, keys, s=("rval", "sum"))
    expected["s2"] = expected["s"] * 2
    _assert_same_rows(ds, expected, keys)


def test_non_vectorized_aggregation(ray_start_regular_shared_2_cpus):
    """``AbsMax`` has no Arrow kernel, so this takes the Python fallback path."""
    joined = _join()
    ds = joined.groupby("id").aggregate(
        Sum(on="rval", alias_name="s"), AbsMax(on="rval", alias_name="am")
    )
    assert _local_aggregates(_physical_dag(ds))

    expected = _pandas_groupby(joined, ["id"], s=("rval", "sum"), am=("rval", "max"))
    _assert_same_rows(ds, expected, ["id"])


def test_empty_join_output(ray_start_regular_shared_2_cpus):
    joined = _join(right=_right().filter(expr=col("rid") > 10_000))
    ds = joined.groupby("id").aggregate(Count(alias_name="c"))
    assert _local_aggregates(_physical_dag(ds))
    assert ds.take_all() == []


def test_global_aggregation_keeps_shuffle(ray_start_regular_shared_2_cpus):
    joined = _join()
    ds = joined.groupby(None).aggregate(Sum(on="rval", alias_name="s"))
    assert _has_aggregate_shuffle(_physical_dag(ds))
    assert ds.take_all() == [{"s": joined.to_pandas()["rval"].sum()}]


def test_legacy_hash_shuffle_is_untouched(
    ray_start_regular_shared_2_cpus, shuffle_v2: DataContext
):
    shuffle_v2.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
    joined = _join()
    ds = joined.groupby("id").aggregate(Sum(on="rval", alias_name="s"))
    assert isinstance(_physical_dag(ds), HashAggregateOperator)
    _assert_same_rows(ds, _pandas_groupby(joined, ["id"], s=("rval", "sum")), ["id"])


def test_complete_aggregation_block_fn():
    """Combines every block it's given into one row per group."""
    aggregate = _make_complete_aggregating_block_fn(
        ("k",), (Sum(on="v", alias_name="s"), Count(alias_name="c"))
    )
    blocks = [
        pa.table({"k": [1, 2], "v": [1, 2]}),
        pa.table({"k": pa.array([], pa.int64()), "v": pa.array([], pa.int64())}),
        pd.DataFrame({"k": [2, 3], "v": [10, 20]}),
        pa.table({"k": [1], "v": [100]}),
    ]
    ctx = TaskContext(task_idx=0, op_name="test")
    out = pa.concat_tables(aggregate(iter(blocks), ctx))
    assert out.sort_by("k").to_pydict() == {
        "k": [1, 2, 3],
        "s": [101, 12, 20],
        "c": [2, 2, 1],
    }
    # No rows means no groups and no output block.
    assert list(aggregate(iter(blocks[1:2]), ctx)) == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
