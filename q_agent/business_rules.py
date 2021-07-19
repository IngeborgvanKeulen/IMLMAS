from datetime import timedelta

from fintel_engine.business_rules.aggregates.models import (
    GroupbyAggregate,
    GroupFn,
    TrxFieldAggregate,
)
from fintel_engine.business_rules.alert_types.models import MTA, TA
from fintel_engine.business_rules.filters.models import ParameterFilter
from fintel_engine.business_rules.resources.enums.operators import ComparisonOperator
from fintel_engine.business_rules.universal_rules.models import (
    UniversalBusinessRule,
    UniversalBusinessRuleSet,
)


RULES = [
    UniversalBusinessRule(
        alias="large-value",
        alert_type=TA(),
        rule_description="Account receives a transaction with value ≥ {x} within a day",
        data_time_window=timedelta(days=1),
        execution_periodicity=1,
        filters=None,
        aggregates=[TrxFieldAggregate(name="return [amount, transaction_id]", fields=["amount", "transaction_id"])],
        parameters=[
            ParameterFilter(
                name="x", comp_op=ComparisonOperator.GE, attribute="amount", value=5000, sweep_range=(5000, 40000, 10),
            )
        ],
    ),
    UniversalBusinessRule(
        alias="sender-trx-day",
        alert_type=MTA(),
        rule_description="Account sends ≥ {x} transactions within a day",
        data_time_window=timedelta(days=1),
        execution_periodicity=1,
        filters=None,
        aggregates=[
            GroupbyAggregate(
                name="return [count, transactions]",
                groupby_fields=["sender"],
                fn_list=[
                    GroupFn(field="amount", func="count", series_name="count"),
                    GroupFn(field="transaction_id", func="unique"),
                ],
            ),
        ],
        parameters=[ParameterFilter(name="x", comp_op=ComparisonOperator.GE, attribute="count", value=7)],
    ),
    UniversalBusinessRule(
        alias="sender-volume-day",
        alert_type=MTA(),
        rule_description="Account sends transactions with combined total value of ≥ {x} within a day",
        data_time_window=timedelta(days=1),
        execution_periodicity=1,
        filters=None,
        aggregates=[
            GroupbyAggregate(
                name="return [sum, transactions]",
                groupby_fields=["sender"],
                fn_list=[
                    GroupFn(field="amount", func="sum", series_name="sum"),
                    GroupFn(field="transaction_id", func="unique"),
                ],
            ),
        ],
        parameters=[
            ParameterFilter(
                name="x", comp_op=ComparisonOperator.GE, attribute="sum", value=22000, sweep_range=(2500, 20000, 10),
            ),
        ],
    ),
    UniversalBusinessRule(
        alias="sender-volume-week",
        alert_type=MTA(),
        rule_description="Account sends transactions with combined total value of ≥ {x} within 2 days",
        data_time_window=timedelta(days=2),
        execution_periodicity=1,
        filters=None,
        aggregates=[
            GroupbyAggregate(
                name="return [sum, transactions]",
                groupby_fields=["sender"],
                fn_list=[
                    GroupFn(field="amount", func="sum", series_name="sum"),
                    GroupFn(field="transaction_id", func="unique"),
                ],
            ),
        ],
        parameters=[
            ParameterFilter(
                name="x", comp_op=ComparisonOperator.GE, attribute="sum", value=40000, sweep_range=(2500, 20000, 10),
            ),
        ],
    ),
]

RULESET = UniversalBusinessRuleSet(RULES)
