import argparse
import calendar
from datetime import date, datetime
import logging
import os
from typing import List, Optional

import pandas as pd
from fintel_engine.business_rules.universal_rules.models import UniversalBusinessRuleSet
from fintel_util.argparse import valid_dynamic_ref
from pathlib import Path


logger = logging.getLogger(__name__)


def valid_date(date_str: str) -> date:
    """ Validate the given date string and return the parsed date. """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date: {date_str} - use the format 'yyyy-mm-dd'")

    return date


def valid_date_str(date_str: str) -> str:
    """ Validate and return the given date string. """
    return str(valid_date(date_str))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simulation")

    # Other
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out-dir",
                        required=False,
                        default="output/",
                        type=Path,
                        help="Directory for output generation.")

    # Logging arguments
    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument("--no-console", action="store_false", help="if true, no stdout output is generated")
    logging_group.add_argument("--log-level", type=str, default="INFO", help='log level (e.g., "INFO", "DEBUG")')
    logging_group.add_argument(
        "--log-file",
        default="",
        help="Name of the log file. When no file name is given, a log file with the name of"
        "the current entry-point will be created (sys.argv[0]). When no directory name is"
        "given, the log file is automatically placed inside args.out_dir. A log path "
        'like "./file.log" also counts as a directory name.',
    )
    # Log directory
    # logging_group.add_argument(
    #     "--log-dir",
    #     required=False,
    #     type=Path,
    #     help="Directory to save log files to if there is no directory in the log filename.",
    # )
    logging_group.add_argument(
        "--skip-log-file", action="store_false", help="Mutually exclusive parameter to entirely disable a log file."
    )
    logging_group.add_argument(
        "--skip-logstash-file",
        action="store_false",
        help="Mutually exclusive parameter to entirely disable a Logstash file.",
    )

    # What to generate
    output = parser.add_argument_group("Output and output formats")
    output.add_argument(
        "--output-to-generate",
        nargs="+",
        type=str,
        choices=["agg_data", "alerts", "counts", "param_sweep", "data_stats", "all"],
        default=["alerts"],
    )
    output.add_argument(
        "--export-fields",
        type=valid_dynamic_ref,
        default="export_files.FILE_FIELDS",
        help="File containing global dict of export fields for 'transactions', 'merchants' and 'stakeholders' (must be "
        "pythonic import path such as 'resources.export_files_dummy.FILE_FIELDS'",
    )

    # Input directories
    data_args = parser.add_argument_group("Input data paths and options")
    data_args.add_argument(
        "--transactions-raw",
        type=Path,
        default=Path("AMLSim/outputs/ml_agent/"),
        help="Path for the transactions csv file")
    data_args.add_argument(
        "--transactions",
        type=Path,
        default=Path("../br_engine/input/"),
        help="Path for the transactions pickle file")

    # Business rule related arguments
    business_rule_args = parser.add_argument_group("Business rule arguments")
    business_rule_args.add_argument(
        "--business-rule-set",
        type=valid_dynamic_ref,
        default="business_rules.RULESET",
        help="Business rule set that contains the business rules to run (must be pythonic import path such as "
        "'business_rules.default_implemented_rules.DEFAULT_RULES'",
    )
    business_rule_args.add_argument(
        "--subset-rules",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument that if passed reduces the rules found in the provided business rule set to the "
        "provided collection of rule aliases",
    )
    business_rule_args.add_argument(
        "--ignore-execution-periodicity",
        action="store_true",
        help="Optional flag that if passed ignores the execution periodicity of rules, meaning that rules will always "
        "fire no matter the date",
    )

    # Parsing days, either a complete month, or individual days
    parsing_days = parser.add_mutually_exclusive_group(required=False)
    parsing_days.add_argument("--year-month", type=str, help="Year-month to process, in YYYYM(M) notation")
    parsing_days.add_argument(
        "--days-to-process",
        nargs="+",
        type=str,
        help="List of days to process (format=%Y-m-%d,e.g. '2019-05-31'). Default is entire month",
    )
    parser.add_argument(
        "--date",
        type=valid_date_str,
        default=str(date.today()),
        help="Date (yyyy-mm-dd) to perform actions for (clean, engine run etc.)",
    )
    return parser


def parse_args_br(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line interface (CLI) arguments.

    :return: CLI arguments
    """
    parser = build_argparser()
    args = parser.parse_args(args=args)

    if not args.out_dir:
        raise parser.error("the following arguments are required: --out-dir")

    # Output to generate
    if "all" in args.output_to_generate:
        args.output_to_generate = ["agg_data", "alerts", "counts", "param_sweep", "data_stats"]

    # Add timestamp to output dir and set the log file
    args.out_dir = args.out_dir / f'br_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(args.out_dir, exist_ok=True)

    # Determine the subset of rules to calculate. args.business_rule_set is kept as the export requires a set of ALL
    # available rules
    if args.subset_rules is not None:
        rule_set = args.business_rule_set.copy()
        try:
            # Create a new instance of a rule set with a reduced number of rules
            args.current_rule_set = UniversalBusinessRuleSet([rule_set.get_rule(alias) for alias in args.subset_rules])
        except ValueError:
            logger.error(
                f"Provided rule aliases were not found in provided business rule set. "
                f"Rule subset aliases: {args.subset_rules}. "
                f"Rules in rule set: {[rule.alias for rule in rule_set.all_rules]}"
            )
    else:
        args.current_rule_set = args.business_rule_set

    # Get number of lookback days
    args.lookback_days = max([rule.data_time_window.days for rule in args.current_rule_set.all_rules])

    # Determine days to calculate
    if args.days_to_process:
        args.days_to_process = sorted(
            [datetime.strptime(day, "%Y-%m-%d").date() for day in args.days_to_process]
        )
    else:
        year, month = int(args.year_month[:4]), int(args.year_month[4:])
        args.days_to_process = list(
            pd.date_range(
                start=date(year, month, 1), end=date(year, month, calendar.monthrange(year, month)[1])
            ).date
        )
    logger.debug(f"Running alert generation from {args.days_to_process[0]} to {args.days_to_process[-1]}")

    return args


def parse_args_transform(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line interface (CLI) arguments.

    :return: CLI arguments
    """
    parser = build_argparser()
    args = parser.parse_args(args=args)

    return args
