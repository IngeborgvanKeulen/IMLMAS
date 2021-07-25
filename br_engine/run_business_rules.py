import json
import logging
import os
import sys
import shutil
import tempfile
import zipfile
from itertools import chain
from typing import List, Optional, Tuple

import pandas as pd
from fintel_engine.business_rules.universal_rules.models import UniversalBusinessRule
from fintel_engine.plotting import DataStatResult
from fintel_engine.transaction_monitoring.rule_runners import run_multiple_days
from fintel_engine.util.encoding import AlertEncoder
from fintel_engine.util.io import save_pickle
from fintel_engine.util.itertools import batch
from fintel_util.logging import log_cmdline_args, setup_logging_module
from mpire import WorkerPool

from br_engine.util import parse_args_br as parse_args

logger = logging.getLogger(__name__)


def write_params(rules: List[UniversalBusinessRule], f) -> None:
    """ Write parameters to file """
    for rule in rules:
        params = "; ".join([repr(param) for param in rule.parameters])
        f.write(f"{rule.alias:20s}  {params}\n")


def prep_transaction_df(trx_df) -> pd.DataFrame:
    """ Helper function to ensure that the provided transaction dataframe has the 'paid_at' attribute available """
    try:
        trx_df.paid_at.min()
    except AttributeError:
        # If not, it's probably the index. If so, reset the index back into a column.
        if trx_df.index.name == "paid_at":
            trx_df.reset_index(inplace=True, drop=False)
        # Else, no required paid_at to be found.
        else:
            raise
    return trx_df


def _batch_create_extended_alerts(shared_objects: Tuple[pd.DataFrame], rule_alert_tuples: list) -> list:
    """ Transforms a batch of BaseAlerts per rule to ExtendedAlerts per rule"""
    transactions_df = prep_transaction_df(shared_objects[0])

    return [
        (rule, [
            {
                "rule": alert.rule.alias,
                "transactions": alert.transactions,
                "date": (transactions_df.paid_at.min(), transactions_df.paid_at.max()),
                "sender": set(transactions_df.sender),
                "receiver": set(transactions_df.receiver),
                "total_amount": transactions_df.amount.sum(),
                "description": alert.rule.rule_description,
            } for alert in alerts
        ]) for rule, alerts in rule_alert_tuples
    ]


def write_alerts(args, results, transactions):
    data_stats_in_result = []
    for day, day_results in results.items():
        final_output_dir = os.path.join(args.out_dir, day.strftime("%Y-%m-%d"))
        os.makedirs(final_output_dir)

        for result_type, res in day_results.items():
            # Extract data_stats if present in result for later concatenation.
            if result_type == "data_stats":
                data_stats_in_result += [DataStatResult(rule, day, data) for rule, data in res.items()]
            else:
                # Transform BaseAlerts to ExtendedAlerts
                if result_type == "alerts":
                    # Transform in parallel
                    with WorkerPool(args.n_jobs) as wp:
                        wp.set_shared_objects((transactions,))
                        tuple_list = wp.map_unordered(
                            _batch_create_extended_alerts,
                            [(b,) for b in batch(list(res.items()))],
                            progress_bar=not args.no_console,
                        )
                    # Update stored alerts
                    res = dict(chain.from_iterable(tuple_list))
                    results[day]["alerts"] = res
                save_pickle(res, os.path.join(final_output_dir, f"{result_type}"))

    file_dir = os.path.join(args.out_dir, "files")

    logger.info("Write alerts to file")
    # Collect alerts over all days
    alerts = [
        dict(alert)
        for results_per_type in results.values()
        for alerts_list in results_per_type["alerts"].values()
        for alert in alerts_list
        if alert["date"] != (pd.NaT, pd.NaT)
    ]
    os.makedirs(file_dir, exist_ok=True)
    zip_path = os.path.join(file_dir, "alert_engine_output.zip")
    zip_dir = tempfile.mkdtemp()
    trx_ids = set(chain.from_iterable(a["transactions"] for a in alerts))

    # Save the days on which the business rules were run
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zfd:
            # Save information of relevant transactions to json
            tx_info = transactions[transactions.transaction_id.isin(trx_ids)][args.export_fields]
            tx_info_name = "tx_info.json"
            tx_info_path = os.path.join(zip_dir, tx_info_name)
            tx_info.reset_index(inplace=True)
            tx_info.to_json(tx_info_path, orient="records")
            zfd.write(tx_info_path, tx_info_name)

            # Save information of all alerts to json
            alert_info_name = "alert_info.json"
            alert_info_path = os.path.join(zip_dir, alert_info_name)
            with open(alert_info_path, "w") as fh:
                json.dump(alerts, fh, cls=AlertEncoder)
            zfd.write(alert_info_path, alert_info_name)
    finally:
        shutil.rmtree(zip_dir)

    logger.info(f"Wrote engine ingest package to {zip_path}")


def main(args: Optional[List[str]] = None, setup_logging: bool = True) -> None:
    args = parse_args(args=args)

    # Perform engine run
    if setup_logging:
        setup_logging_module(args)
        log_cmdline_args(args, exclude=["daily_rules", "weekly_rules", "monthly_rules"])

    # Log the business rule parameters
    rule_log_file = os.path.join(args.out_dir, "business_rule_params.log")
    with open(rule_log_file, "w") as f:
        f.write("Day rules\n")
        write_params(args.current_rule_set.day_rules, f)
        f.write("\nWeek rules\n")
        write_params(args.current_rule_set.week_rules, f)
        f.write("\nMonth rules\n")
        write_params(args.current_rule_set.month_rules, f)

    # Log the output and log paths
    logger.info("Starting alert generation")
    logger.info(f"{'Output:':15s} {args.out_dir}")
    logger.info(f"{'Logs:':15s} {args.log_file}")
    logger.info(f"{'Parameters:':15s} {rule_log_file}")

    logger.info("Loading and preparing all dataframes")
    transactions, merchants, refunds, chargebacks = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    transactions = pd.read_pickle(os.path.join(args.transactions, "transactions.pkl"))
    transactions.index = pd.to_datetime(transactions.index) + pd.DateOffset(hours=12)

    logger.info("Running business rules")
    os.makedirs(args.out_dir, exist_ok=True)

    output_to_generate = args.output_to_generate or {"alerts", "counts"}

    logger.info("Running alert generation")
    results = run_multiple_days(
        days=args.days_to_process,
        business_rule_set=args.current_rule_set,
        transactions=transactions,
        refunds=refunds,
        chargebacks=chargebacks,
        merchants=merchants,
        stats=None,
        peer_group_api=None,
        output_to_generate=output_to_generate,
        ignore_execution_periodicity=args.ignore_execution_periodicity,
        progress_bar=not args.no_console,
        n_jobs=args.n_jobs,
    )

    write_alerts(args, results, transactions)


if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)
    if argc < 1:
        print("Usage: python3 %s [ConfJSON]" % argv[0])
        exit(1)

    conf_file = argv[1]
    with open(conf_file, "r") as rf:
        conf = json.load(rf)

    days = conf["days_to_process"]
    main(args=["--days-to-process"] + days)
