"""
run_business_rules expects a transaction dataframe, sorted on index. The index is the paid_at date of the transaction.
it should have the columns "amount" and "transaction_id"
"""
import datetime
import json
import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from q_agent.process_alerts import main as process_alerts


def transform_transactions(trx_path: str, output_path: str):
    # Load in the transaction csv file
    transactions_df = pd.read_csv(os.path.join(trx_path, "transactions.csv"), header=0)

    # Rename columns to the name the engine expects
    transactions_df = transactions_df.rename(columns={"tran_id": "transaction_id",
                                                      "tran_timestamp": "paid_at",
                                                      "base_amt": "amount",
                                                      "orig_acct": "sender",
                                                      "bene_acct": "receiver"
                                                      })

    # Set the paid_at column as the index and sort the dataframe based on this
    transactions_df.set_index("paid_at", inplace=True)
    transactions_df.sort_index(inplace=True)

    # Write the transformed dataframe as a pickle
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_path = os.path.join(output_path, "transactions.pkl")
    if not os.path.exists(file_path):
        transactions_df.to_pickle(file_path)
    else:
        # Combine with existing dataframe if exists
        existing_df = pd.read_pickle(file_path)

        # Replace duplicate transaction ids with something else
        existing_df["transaction_id"] += len(transactions_df)

        combined_df = pd.concat([existing_df, transactions_df])
        combined_df.to_pickle(file_path)


def main(brs: List[int]) -> None:
    # Transform transactions such that the business rules can be ran
    trx_path = Path("../AMLSim/outputs/ml_agent/")
    output_path = Path("../br_engine/input/")
    transform_transactions(trx_path, output_path)

    # Run business rules
    conf_file = "./amlsim_conf.json"
    with open(conf_file, "r") as rf:
        conf = json.load(rf)
    start_date = datetime.datetime.strptime(conf["general"]["start_date"], "%Y-%m-%d")
    current_date = datetime.datetime.strptime(conf["general"]["base_date"], "%Y-%m-%d")
    end_date = current_date + datetime.timedelta(days=int(conf["general"]["total_steps"]))
    days = end_date - start_date
    days = [start_date + datetime.timedelta(days=i) for i in range(days.days)]
    days = [i.strftime("%Y-%m-%d") for i in days]

    subset_rules = np.array(["large-value", "sender-trx-day", "sender-volume-day", "sender-volume-week"])

    # todo: implement or connect your own business rules logic here (ours needed the days and rule options)
    # The rest of the code expects a zip written to the output path defined in the engine_conf.json file. This zip
    # should contain a json file called tx_info.json, containing a dict per transaction that has triggered an alert.

    # Process the alerts such that we have feedback for the agents and an evaluation for the business rules
    conf_file = "./engine_conf.json"
    with open(conf_file, "r") as rf:
        conf = json.load(rf)

    conf = conf["process_alerts"]
    process_alerts(conf=conf)

    # Clean up all output except for transactions
    os.system("rm -r ./output/*")
    bashCommand = "AMLSim/scripts/clean_logs.sh"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd="../")
    _, _ = process.communicate()


if __name__ == "__main__":
    main(brs=[3])
