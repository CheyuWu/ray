from typing import List, Optional, Type

from ray.data._internal.compute import TaskPoolStrategy
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.hash_aggregate_v2 import (
    _make_complete_aggregating_block_fn,
)
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.execution.operators.shuffle_operators.disk_shuffle_map_operator import (  # noqa: E501
    DiskHashShuffleMapOp,
)
from ray.data._internal.execution.operators.shuffle_operators.disk_shuffle_reduce_operator import (  # noqa: E501
    DiskHashShuffleReduceOp,
)
from ray.data._internal.execution.operators.shuffle_operators.shuffle_map_operator import (  # noqa: E501
    ShuffleMapOp,
)
from ray.data._internal.execution.operators.shuffle_operators.shuffle_reduce_operator import (  # noqa: E501
    ShuffleReduceOp,
)
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators import Aggregate
from ray.data._internal.logical.rules.operator_fusion import FuseOperators
from ray.data._internal.output_buffer import OutputBlockSizeOption

__all__ = ["EliminateRedundantShuffles"]

_SHUFFLE_MAP_OPS = (ShuffleMapOp, DiskHashShuffleMapOp)
_SHUFFLE_REDUCE_OPS = (ShuffleReduceOp, DiskHashShuffleReduceOp)


class EliminateRedundantShuffles(Rule):
    """Drops the shuffle of a group-by whose input is already partitioned as the
    group-by requires, e.g. by a join on a subset of the group-by keys::

        JoinShuffleReduce -> HashAggregateMap -> HashAggregateReduce   # before
        JoinShuffleReduce -> HashAggregate(mode=complete)              # after

    The join's reduce then emits each partition as one bundle, so every task of
    the one-phase aggregation sees whole groups. Runs after ``FuseOperators``,
    as only maps fused into that reduce keep partitions whole.
    """

    @classmethod
    def dependencies(cls) -> List[Type[Rule]]:
        return [FuseOperators]

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        self._op_map = plan.op_map.copy()
        dag = self._eliminate_shuffles_in_dag(plan.dag)
        return PhysicalPlan(dag, self._op_map, plan.context)

    def _eliminate_shuffles_in_dag(self, op: PhysicalOperator) -> PhysicalOperator:
        # Upstream first. A one-phase aggregation is a map, so a group-by right
        # after it keeps its shuffle.
        op._input_dependencies = [
            self._eliminate_shuffles_in_dag(input_op)
            for input_op in op.input_dependencies
        ]
        local_op = self._try_plan_local_aggregate(op)
        if local_op is None:
            return op
        return local_op

    def _try_plan_local_aggregate(
        self, reduce_op: PhysicalOperator
    ) -> Optional[MapOperator]:
        """Replace ``reduce_op`` and its map op, the two halves of a group-by's
        shuffle, with a one-phase aggregation if the shuffle is redundant."""
        aggregate = self._op_map.get(reduce_op)
        if not (
            isinstance(reduce_op, _SHUFFLE_REDUCE_OPS)
            and isinstance(aggregate, Aggregate)
        ):
            return None
        (map_op,) = reduce_op.input_dependencies
        if not isinstance(map_op, _SHUFFLE_MAP_OPS):
            return None
        (source,) = map_op.input_dependencies
        # Only a hash-shuffle reduce can hand over whole partitions.
        if not isinstance(source, _SHUFFLE_REDUCE_OPS):
            return None

        distribution = aggregate.required_input_distribution
        partitioning = source.output_partitioning
        if distribution is None or not distribution.is_satisfied_by(partitioning):
            return None
        # An explicit ``groupby(num_partitions=...)`` asks for that parallelism.
        if (
            aggregate.num_partitions is not None
            and aggregate.num_partitions != partitioning.num_partitions
        ):
            return None

        key_columns = distribution.columns
        transformer = MapTransformer(
            [
                BlockMapTransformFn(
                    _make_complete_aggregating_block_fn(
                        key_columns, tuple(aggregate.aggs)
                    ),
                    output_block_size_option=OutputBlockSizeOption.of(
                        target_max_block_size=(
                            reduce_op.data_context.target_max_block_size
                        )
                    ),
                )
            ]
        )
        name = f"HashAggregate(key_columns={key_columns}, mode=complete)"
        # Keep any map that ``FuseOperators`` fused into the aggregation's reduce.
        if reduce_op._fused_output_map_transformer is not None:
            transformer = transformer.fuse(reduce_op._fused_output_map_transformer)
            # Not parsed from ``reduce_op.name``: key names may contain ``->``.
            fused_map_names = [op.name for op in reduce_op._logical_operators[1:]]
            name = "->".join([name, *fused_map_names])

        source.set_emit_whole_partitions()
        local_op = MapOperator.create(
            transformer,
            source,
            reduce_op.data_context,
            target_max_block_size_override=(
                reduce_op._fused_output_map_target_max_block_size_override
            ),
            name=name,
            compute_strategy=TaskPoolStrategy(),
            # Keep ``FuseOperators`` from folding it into ``source``'s reduce.
            supports_fusion=False,
            map_task_kwargs=reduce_op._fused_output_map_task_kwargs,
        )
        local_op.set_logical_operators(*reduce_op._logical_operators)

        # ``local_op`` already reads from ``source``; unlink the shuffle's ops.
        source._output_dependencies.remove(map_op)
        local_op._output_dependencies = reduce_op.output_dependencies
        self._op_map.pop(map_op)
        self._op_map.pop(reduce_op)
        self._op_map[local_op] = aggregate
        return local_op
