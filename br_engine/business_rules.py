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
        rule_description="Merchant receives a transaction with value ≥ {x} within a day",
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
        alias="value-same-person-week",
        alert_type=MTA(),
        rule_description="Person receives more than {y} transactions with combined total value of ≥ {x} within a week"
        " from the same sender",
        data_time_window=timedelta(days=7),
        execution_periodicity=4,
        filters=None,
        aggregates=[
            GroupbyAggregate(
                name="return [sum, transactions, count]",
                groupby_fields=["receiver", "sender"],
                fn_list=[
                    GroupFn(field="amount", func="sum", series_name="sum"),
                    GroupFn(field="transaction_id", func="unique"),
                    GroupFn(field="amount", func="count", series_name="count"),
                ],
            ),
        ],
        parameters=[
            ParameterFilter(
                name="x", comp_op=ComparisonOperator.GE, attribute="sum", value=8000, sweep_range=(2500, 20000, 10),
            ),
            ParameterFilter(name="y", comp_op=ComparisonOperator.GE, attribute="count", value=3,),
        ],
    ),
    UniversalBusinessRule(
        alias="same-person-week",
        alert_type=MTA(),
        rule_description="Person receives ≥ {x} transactions from the same sender "
        "within a week",
        data_time_window=timedelta(days=7),
        execution_periodicity=4,
        filters=None,
        aggregates=[
            GroupbyAggregate(
                name="return [count_account, transactions]",
                groupby_fields=["receiver", "sender"],
                fn_list=[
                    GroupFn(field="amount", func="count", series_name="count"),
                    GroupFn(field="transaction_id", func="unique"),
                ],
            ),
        ],
        parameters=[
            ParameterFilter(
                name="x", comp_op=ComparisonOperator.GE, attribute="count", value=5, sweep_range=(2, 10, 10),
            )
        ],
    ),
    UniversalBusinessRule(
        alias="same-person-day",
        alert_type=MTA(),
        rule_description="Person receives ≥ {x} transactions from the same sender within a day",
        data_time_window=timedelta(days=1),
        execution_periodicity=1,
        filters=None,
        aggregates=[
            GroupbyAggregate(
                name="return [count, transactions]",
                groupby_fields=["receiver", "sender"],
                fn_list=[
                    GroupFn(field="amount", func="count", series_name="count"),
                    GroupFn(field="transaction_id", func="unique"),
                ],
            ),
        ],
        parameters=[ParameterFilter(name="x", comp_op=ComparisonOperator.GE, attribute="count", value=3)],
    ),
]

RULESET = UniversalBusinessRuleSet(RULES)
