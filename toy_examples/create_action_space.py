from typing import Dict

import numpy as np


def business_rule_1(amount: int, use_br: bool, threshold: int) -> bool:
    """
    Account receives a transaction with value >= threshold within a day.

    :param amount: transaction value
    :param use_br: whether to use this br
    :param threshold: the threshold
    :return: True when the laundering was successful, false when it is caught
    """
    if not use_br:
        return True

    if amount < threshold:
        return True
    else:
        return False


def business_rule_2(account: int, use_br: bool, threshold: int) -> bool:
    """
    Account sends >= threshold transactions within a day. Only works for scatter-gather.

    :param account: number of involved accounts
    :param use_br: whether to use this br
    :param threshold: the threshold
    :return: True when the laundering was successful, false when it is caught
    """
    if not use_br:
        return True

    if (account - 2) < threshold:
        return True
    else:
        return False


def business_rule_3(amount: int, account: int, use_br: bool, threshold: int) -> bool:
    """
    Account sends transactions with combined total value >= threshold within a day. Only works for scatter-gather.

    :param amount: transaction value
    :param account: number of involved accounts
    :param use_br: whether to use this br
    :param threshold: the threshold
    :return: True when the laundering was successful, false when it is caught
    """
    if not use_br:
        return True

    if amount * (account - 2) < threshold:
        return True
    else:
        return False


def create_action_space(params: Dict, div_amount: int, div_account: int, div_days: int, div_reward, brs: Dict)\
        -> np.array:
    """
    Create the action space

    :param params: contains the minimums and stepsizes
    :param div_amount: the normalization value for amount and therefore as well the max value
    :param div_account: the normalization value for account and therefore as well the max value
    :param div_days: the normalization value for days and therefore as well the max value
    :param brs: dict with business rules, their thresholds and whether they should be used
    :return: the action space
    """
    # Set the defaults
    success = params["success"]
    caught = params["caught"]
    action_space = None

    # Create the action space
    for amount in range(params["min_amount"], div_amount + 1, params["step_amount"]):
        for account in range(params["min_account"], div_account + 1, params["step_account"]):
            for day in range(params["min_days"], div_days + 1, params["step_days"]):
                if business_rule_1(amount=amount, **brs[1]) and business_rule_2(account=account, **brs[2]) and \
                        business_rule_3(amount=amount, account=account, **brs[3]):
                    t = np.array([amount / div_amount, account / div_account, day / div_days, success])
                else:
                    if amount == 0:
                        t = np.array([amount / div_amount, account / div_account, day / div_days, success])
                    else:
                        t = np.array([amount / div_amount, account / div_account, day / div_days, caught])
                try:
                    action_space = np.vstack((action_space, np.hstack((t))))
                except:
                    action_space = t

    return action_space
