import glob
import json
import os
import pickle
import sys
import zipfile
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd


def full_investigation(exp_trx_ids: pd.DataFrame, trx_ids: Set) -> Tuple[int, int, int, Set]:
    """
    In this function we assume the investigation team performs a full investigation. In other words, if one of the
    transactions in a ML network is found, the complete network is found.

    :param exp_trx_ids: The laundering transaction ids from the laundering agent
    :param trx_ids: All transaction ids that triggered an alert
    :return: the # TP, # FP, # FN and which valid transactions were found
    """
    # Create dicts from alert to trx and vice-versa
    alert_to_trx = dict(exp_trx_ids.groupby("alert_id")["tran_id"].apply(list))
    trx_to_alert = dict(zip(exp_trx_ids["tran_id"], exp_trx_ids["alert_id"]))

    # Find which complete networks (aka alerts) have been found
    alerts_found = set([trx_to_alert.get(tr, None) for tr in trx_ids]) - {None}

    # We need the number of transactions that were in the found networks for further calculations
    trx_found = [alert_to_trx[alert_id] for alert_id in alerts_found]
    trx_found = set([item for sublist in trx_found for item in sublist])

    # Calculate the false positive, true positive and false negative
    tp = len(trx_found)
    fp = len(trx_ids - trx_found)
    fn = len(trx_to_alert.keys()) - tp

    return tp, fp, fn, trx_found


def simple_investigation(exp_trx_ids: pd.DataFrame, trx_ids: Set) -> Tuple[int, int, int, Set]:
    """
    In this function we assume the investigation team performs a simple investigation. In other words, if the found
    transactions are not the transactions involving the main account, then the network is not found.

    :param exp_trx_ids: The laundering transaction ids from the laundering agent
    :param trx_ids: All transaction ids that triggered an alert
    :return: the # TP, # FP, # FN and which valid transactions were found
    """
    # Get the transaction ids that need to be found
    exp_trx_ids = set(exp_trx_ids["tran_id"].values)

    # Calculate the false positive, true positive and false negative
    fp = trx_ids - exp_trx_ids
    tp = trx_ids.intersection(exp_trx_ids)
    fn = exp_trx_ids - trx_ids

    return len(tp), len(fp), len(fn), trx_ids


def get_performance_aml(conf: Dict, exp_trx_ids: pd.DataFrame, trx_ids: set) -> \
        Tuple[Set, float, float, float, int, int, int]:
    """
    Evaluate the performance of the AML method

    :param conf: a conf containing some parameter settings
    :param exp_trx_ids : the expected transaction ids to be detected per alert id in a dataframe
    :param trx_ids : the found transaction ids by the aml method
    :return: the transactions that were tp and found, precision, recall and f1-score
    """
    if conf["full_investigation"]:
        tp, fp, fn, trx_found = full_investigation(exp_trx_ids, trx_ids)
    else:
        tp, fp, fn, trx_found = simple_investigation(exp_trx_ids, trx_ids)

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    try:
        f1_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0.0

    return trx_found, precision, recall, f1_score, tp, fp, fn


def get_feedback(alert_df: pd.DataFrame, trx_ids: set) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """
    :param alert_df : the dataframe containing the sars
    :param trx_ids : the found transaction ids by the aml method

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

        received_pp = subset.groupby("bene_acct")["base_amt"].sum()
        send_pp = subset.groupby("orig_acct")["base_amt"].sum()
        if alert_type == "scatter_gather":
            reward = max(send_pp)
        elif alert_type == "cycle":
            reward = min(received_pp)
        else:
            raise ValueError(f"alert_type of {alert_type} has not been implemented yet")

        total_prevented = reward if True in alert_df["found"].values else 0
        feedback[alert_type] = (reward - total_prevented, reward)

    return feedback, total_prevented


def get_data(conf: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """
    Get all data
    :param conf: conf that contains parameter settings
    :return: dataframes containing the main account, alerts, transactions and combination of alert and transaction id,
    and a set containing the found transaction ids
    """
    # Get the last alert generation run
    dir = sorted(glob.glob(os.path.join(conf["output_path"], "*")))[-1]
    archive = zipfile.ZipFile(dir + conf["engine_zip"], "r")

    # Read in the output of the alert generation
    trx_info = pd.read_json(archive.read("tx_info.json"))

    # Get the main account ids
    main_df = pd.read_csv(os.path.join(conf["input_path"], "main_accounts.csv"), header=0)

    # Get the expected transaction ids per alert and the found transaction ids by the AML method
    alert_df = pd.read_csv(os.path.join(conf["input_path"], "alert_transactions.csv"), header=0)
    exp_trx_ids = alert_df[["alert_id", "tran_id"]]

    if len(trx_info) == 0:
        trx_ids = set()
    else:
        trx_ids = set(trx_info["transaction_id"].values)

    return main_df, alert_df, trx_info, exp_trx_ids, trx_ids


def main(conf: Dict = None):
    """
    Main function to process alerts into feedback and evaluation of AML

    :param conf: conf containing parameter settings
    """
    # Get the data
    main_df, alert_df, trx_info, exp_trx_ids, trx_ids = get_data(conf=conf)

    # Get the performance of the AML method for this iteration
    trx_found, prec, rec, f1, tp, fp, fn = get_performance_aml(conf, exp_trx_ids, trx_ids)

    # Get feedback for the agent
    feedback, prevented = get_feedback(alert_df, trx_found)
    laundered = sum([x[0] for x in feedback.values()])

    if not os.path.exists(conf["result_path"]):
        os.makedirs(conf["result_path"])

    file_path = os.path.join(conf["result_path"], conf["result_file"])
    file_exists = os.path.isfile(file_path)

    n_trx = len(pd.read_csv(conf["transactions_path"]))
    use_columns = ["trx", "tp", "fp", "fn", "recall", "precision", "f1-score", "prevented", "laundered"]
    if not file_exists:
        df = pd.DataFrame(np.array([[n_trx, tp, fp, fn, rec, prec, f1, prevented, laundered]]), columns=use_columns)
    else:
        df = pd.read_csv(file_path)
        new_df = pd.DataFrame(np.array([[n_trx, tp, fp, fn, rec, prec, f1, prevented, laundered]]), columns=use_columns)
        df = df.append(new_df)

    df.to_csv(file_path, index=False)

    with open('feedback.pkl', 'wb') as handle:
        pickle.dump(feedback, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)
    if argc < 1:
        print("Usage: python3 %s [ConfJSON]" % argv[0])
        exit(1)

    conf_file = argv[1]
    with open(conf_file, "r") as rf:
        conf = json.load(rf)

    conf = conf["process_alerts"]
    main(conf=conf)
