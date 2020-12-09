import os
import pickle
import random
import time
from typing import List, Optional

import numpy as np
import pandas as pd


ACTION_TYPES = ["accounts", "amount", "period"]
LIMITS = {
    "accounts": np.arange(4, 51),
    "amount": np.arange(100, 1000001, 100),  # 0.04s for 500, 0.07s for 100, 0.4 for 10
    "period": np.arange(1, 31)
}


def write_features(features: pd.DataFrame, new_features: List):
    """
    This function writes the feature values used by the simulation to a file
    """
    for idx, value in enumerate(new_features):
        features.iloc[0, idx+3] = value
    features.to_csv("../AMLSim/paramFiles/ml_agent/alertPatterns.csv", index=False)


def read_features() -> pd.DataFrame:
    """
    This function reads the feature values used in the last simulation run/step.
    """
    parameters = pd.read_csv("../AMLSim/paramFiles/ml_agent/alertPatterns.csv")
    return parameters


def process_feedback(weights: np.array, features: pd.DataFrame, discount: float, lr: float) -> np.array:
    """
    This function updates the weights based on the feedback obtained in the last simulation step
    """
    # Get the feedback and process it
    with open("./feedback.pkl", "rb") as handle:
        feedback = pickle.load(handle)

    # for alert_type, values in feedback:
    for idx, row in features.iterrows():
        alert_type = row["type"]
        reward_obt = feedback[alert_type][0]
        reward_tot = feedback[alert_type][1]
    reward = reward_obt

    # Update the weights and save them
    possible_states = np.array(get_reachable_states(features))
    max_possible_expected_return = max(np.dot(possible_states, weights))
    features = features.iloc[0, 3:-2].values
    current_q = np.dot(features, weights)
    weights = weights + lr * (reward + discount * max_possible_expected_return - current_q) * features
    np.save("./weights.npy", weights)

    return weights


def get_reachable_states(features: pd.DataFrame, explore=False):
    """
    This function returns the reachable states from the given state.
    In other words, the reachable feature values from the given feature values.
    We have limited this by only being able to change one feature value.

    If explore is true, then we first have to randomly select an action type to ensure each action type has the same
    possibility of being chosen.
    """
    if explore:
        action_options = [random.choice(ACTION_TYPES)]
    else:
        action_options = ACTION_TYPES

    reachable = set()
    for action in action_options:
        # Get the old setting of the feature values
        old_features = list(features.iloc[0].values)

        # Find the index of the feature we want to change as action
        action_idx = np.where(features.columns == "min_" + action)[0][0]

        # Add the new options as reachable states
        for value in LIMITS[action]:
            old_features[action_idx] = value
            old_features[action_idx + 1] = value
            reachable.add(tuple(old_features[3:-2]))

    # To ensure no double states are in the set first transform it to a set and then back to a list
    return list(reachable)


def main(args: Optional[List[str]] = None) -> None:
    step = 0  # todo: get as input the step such that we can obtain the current lr and epsilon value
    print(f"Run step {step} for the reinforcement agent")

    # Set the parameters
    epsilon = pow(0.95, step)
    lr = 0.1
    discount = 0.5
    n_features = len(ACTION_TYPES) * 2

    # Get the features used in the last step of the simulation, as well as the corresponding weights and feedback
    features = read_features()
    if os.path.exists("./weight.npy"):
        weights = np.load("./weight.npy")
    else:
        # Initialize the weights optimistically in order to encourage exploration
        weights = np.ones(n_features)

    # Process the feedback by updating the weights
    weights = process_feedback(weights, features, discount, lr)

    # Select either the most optimal action, or random
    if random.random() < epsilon:
        # Explore; Get new_features that have been randomly obtained
        new_features = random.choice(get_reachable_states(features, explore=True))
    else:
        # Exploit; Get new_features that give max expected return
        possible_states = np.array(get_reachable_states(features))
        new_features = possible_states[np.argmax(np.dot(possible_states, weights))]

    # Save the new feature values
    write_features(features, new_features)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Time: ", end-start)
