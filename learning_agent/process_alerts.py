import glob
import os
import pickle
import zipfile
from typing import Dict, List, Tuple, Optional

import pandas as pd


def get_performance_aml(exp_trx_ids: set, trx_ids: set):
    """
    :param exp_trx_ids : the expected transaction ids to be detected
    :param trx_ids : the found transaction ids by the aml method
    """
    fp = trx_ids - exp_trx_ids
    tp = trx_ids.intersection(exp_trx_ids)
    fn = exp_trx_ids - trx_ids

    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    print("Precision: ", precision)
    print("Recall: ", recall)

    if precision + recall == 0:
        print("F1-score: 0")
    else:
        print("F1-score: ", 2 * precision * recall / (precision + recall))


def get_feedback(alert_df: pd.DataFrame, trx_ids: set, main_df: pd.DataFrame) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """
    :param alert_df : the dataframe containing the sars
    :param trx_ids : the found transaction ids by the aml method
    :param main_df: the info of the main accounts

    :return a dict with as key the typology and the value the reward, and the total prevented money
    """
    # Add an extra column to denote if a transaction has been found
    alert_df["found"] = alert_df["tran_id"].isin(trx_ids)

    # Loop over all different AML typologies
    typologies = list(set(alert_df["alert_type"].values))
    feedback = {}
    total_prevented = 0
    for alert_type in typologies:
        # First find the maximum possible reward for each type
        subset = alert_df[alert_df["alert_type"] == alert_type]
        main_acc = main_df[main_df["ALERT_ID"] == subset["alert_id"].values[0]]["MAIN_ACCOUNT_ID"]
        sender = sum(subset[subset["orig_acct"].isin([main_acc])]["base_amt"].values)
        receiver = sum(subset[subset["bene_acct"].isin([main_acc])]["base_amt"])
        reward = max(sender, receiver)

        # Now subtract all transactions that have been detected by the AML method
        field = "bene_acct"
        if reward == sender:
            field = "orig_acct"

        # Don't fall for the == is the same as is, as it isn't
        prevented = sum(subset[(subset[field].isin([main_acc])) & (subset["found"] == True)]["base_amt"].values)

        feedback[alert_type] = (reward - prevented, reward)
        total_prevented += prevented

    return feedback, total_prevented


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, set, set]:
    # todo: make conf / argument for the dir and archive file place
    input_path = "../AMLSim/outputs/ml_agent"

    # Get the last alert generation run
    dir = sorted(glob.glob("../output/*"))[-1]
    archive = zipfile.ZipFile(dir + "/files/alert_engine_output.zip", "r")

    # Read in the output of the alert generation
    trx_info = pd.read_json(archive.read("tx_info.json"))

    # Get the main account ids
    main_df = pd.read_csv(os.path.join(input_path, "main_accounts.csv"), header=0)
    main_accounts = main_df["MAIN_ACCOUNT_ID"].values

    alert_df = pd.read_csv(os.path.join(input_path, "alert_transactions.csv"), header=0)
    # Keep certain rows with the interesting transactions (including main account, sender or receiver depends on type)
    alert_df = alert_df[
        alert_df["orig_acct"].isin(main_accounts) | alert_df["bene_acct"].isin(main_accounts)
    ]

    # Get the expected transaction ids (the truly suspicious ones) and the found transaction ids by the AML method
    exp_trx_ids = set(alert_df["tran_id"].values)
    trx_ids = set(trx_info["transaction_id"].values)

    return main_df, alert_df, exp_trx_ids, trx_ids


def main(args: Optional[List[str]] = None) -> None:
    # Get the data
    main_df, alert_df, exp_trx_ids, trx_ids = get_data()

    # Get the performance of the AML method for this iteration
    get_performance_aml(exp_trx_ids, trx_ids)

    # Get feedback for the agent
    feedback, prevented = get_feedback(alert_df, trx_ids, main_df)
    print("The total amount of money that has been prevented from being laundered: ", prevented)
    print("The total amount of money that has been successfully laundered: ", sum([x[0] for x in feedback.values()]))

    with open('feedback.pkl', 'wb') as handle:
        pickle.dump(feedback, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
