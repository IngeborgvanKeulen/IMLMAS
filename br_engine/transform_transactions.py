"""
run_business_rules expects a transaction dataframe, sorted on index. The index is the paid_at date of the transaction.
it should have the columns "amount" and "transaction_id"
"""
import os
from typing import List, Optional

import pandas as pd

from br_engine.util import parse_args_transform as parse_args


def transform_transactions(args):
    # Load in the transaction csv file
    transactions_df = pd.read_csv(os.path.join(args.transactions_raw, "transactions.csv"), header=0)

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
    if not os.path.exists(args.transactions):
        os.makedirs(args.transactions)
    transactions_df.to_pickle(os.path.join(args.transactions, "transactions.pkl"))


def main(args: Optional[List[str]] = None) -> None:
    args = parse_args(args=args)
    transform_transactions(args)


if __name__ == "__main__":
    main()
