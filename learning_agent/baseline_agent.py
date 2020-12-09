import glob
import os
import pickle
import random
import zipfile
from typing import Dict, List, Tuple, Optional

import pandas as pd

# These changes are viewed from a safe perspective
CHANGES = {
    "accounts": 1,
    "amount": -1000,
    "period": 1
}

def random_risk_action(alert_type: str, parameter: pd.Series):
    options = {"accounts", "amount", "period"}
    if parameter["min_period"] == 1:
        options = options - {"period"}
    if parameter["min_accounts"] == 3:
        options = options - {"accounts"}
    field = random.choice(list(options))

    # field = "amount"
    template = {"count": parameter["count"],
                "type": parameter["type"],
                "schedule_id": parameter["schedule_id"],
                "min_accounts": parameter["min_accounts"],
                "max_accounts": parameter["max_accounts"],
                "min_amount": parameter["min_amount"],
                "max_amount": parameter["max_amount"],
                "min_period": parameter["min_period"],
                "max_period": parameter["max_period"],
                "bank_id": parameter["bank_id"],
                "is_sar": parameter["is_sar"],
                }

    template["min_" + field] = template["min_" + field] - CHANGES[field]
    template["max_" + field] = template["max_" + field] - CHANGES[field]
    return template


def random_safe_action(alert_type: str, parameter: pd.Series):
    options = {"accounts", "amount", "period"}
    if parameter["min_period"] == 30:
        options = options - {"period"}
    if parameter["min_amount"] + CHANGES["amount"] < 0:
        options = options - {"amount"}
    field = random.choice(list(options))

    # field = "amount"
    template = {"count": parameter["count"],
                "type": parameter["type"],
                "schedule_id": parameter["schedule_id"],
                "min_accounts": parameter["min_accounts"],
                "max_accounts": parameter["max_accounts"],
                "min_amount": parameter["min_amount"],
                "max_amount": parameter["max_amount"],
                "min_period": parameter["min_period"],
                "max_period": parameter["max_period"],
                "bank_id": parameter["bank_id"],
                "is_sar": parameter["is_sar"],
                }

    template["min_" + field] = template["min_" + field] + CHANGES[field]
    template["max_" + field] = template["max_" + field] + CHANGES[field]
    return template


def main(args: Optional[List[str]] = None) -> None:
    # Get the data
    with open('./feedback.pkl', 'rb') as handle:
        feedback = pickle.load(handle)

    parameters = pd.read_csv("../AMLSim/paramFiles/ml_agent/alertPatterns.csv")

    new_parameters = pd.DataFrame(columns=parameters.columns)
    # for alert_type, values in feedback:
    for idx, row in parameters.iterrows():
        alert_type = row["type"]
        reward_obt = feedback[alert_type][0]
        reward_tot = feedback[alert_type][1]

        # If more than half was laundered, consider it a success
        if reward_obt/reward_tot > 0.5:
            # Change the parameters to launder more money
            new_parameters = new_parameters.append(random_risk_action(alert_type, row), ignore_index=True)

        else:
            # Change the parameters to be more safe
            new_parameters = new_parameters.append(random_safe_action(alert_type, row), ignore_index=True)

    new_parameters.to_csv("../AMLSim/paramFiles/ml_agent/alertPatterns.csv", index=False)


if __name__ == "__main__":
    main()
