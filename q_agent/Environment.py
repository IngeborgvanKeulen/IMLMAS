import datetime
import json
import logging
import os
import pickle
import subprocess
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from q_agent.Detection import main as run_detection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BenchmarkEnvironment:
    def __init__(self, action_space: np.array, div_amount: int, div_account: int, div_days: int,
                 div_reward: int, model: str, max_days: int):
        """
        Initialize the environment

        :param action_space: all the possible actions
        :param div_amount: the normalization value for the feature amount
        :param div_account: the normalization value for the feature account
        :param div_days: the normalization value for the feature days
        :param div_reward: the normalization value for the reward
        :param model: which model is used; "DQN", "BOOTSTRAP", "SVM"
        :param max_days: the max number of days the agent can try launder money
        """
        self.action_space = action_space
        self.div_amount = div_amount
        self.div_account = div_account
        self.div_days = div_days
        self.div_reward = div_reward
        self.max_days = max_days
        self.model = model

    def reset(self):
        """
        Reset the complete environment

        :return: the start state
        """
        # Clear all transactions before a new game starts
        os.system("rm -r ../br_engine/input/transactions.pkl")

        return np.array([0 / self.div_amount, 0 / self.max_days])

    def step(self, brs: List[int], state: Union[tf.Tensor, np.array], action: int) \
            -> Tuple[Union[tf.Tensor, np.array], int, bool]:
        """
        Perform the action and return the results of doing this action

        :param brs: the business rules to use
        :param state: the current state
        :param action: the chosen action (index in the action space)
        :return: the next state the agent ended up in, the obtained reward and whether the agent is caught (done = True)
        """
        # Set the defaults
        done = False

        # Get the next state that is obtained by doing action in the current state
        next_state = np.copy(state)
        next_state[0] = np.copy(self.action_space[action][0])
        next_state[1] += int(np.copy(self.action_space[action][2]) * self.div_days) / self.max_days

        # Perform the AML detection method
        # Write the chosen action to the corresponding file
        features = pd.read_csv("../AMLSim/paramFiles/ml_agent/alertPatterns.csv")
        action_selected = self.action_space[action] * [self.div_amount, self.div_account, self.div_days]
        action_selected = [action_selected[1], action_selected[0], action_selected[2]]
        features.iloc[0, 3:9] = [int(i) for i in action_selected for _ in (0, 1)]
        features.to_csv("../AMLSim/paramFiles/ml_agent/alertPatterns.csv", index=False)

        # Update the simulation date and steps
        with open("amlsim_conf.json") as f:
            simulation = json.load(f)
        simulation["general"]["total_steps"] = int(action_selected[2])
        start_date = datetime.datetime.strptime(simulation["general"]["start_date"], "%Y-%m-%d")
        current_date = start_date + datetime.timedelta(days=int(state[1] * self.max_days))
        simulation["general"]["base_date"] = current_date.strftime("%Y-%m-%d")
        json.dump(simulation, open("amlsim_conf.json", "w"), indent=2)

        # Run the simulation part
        bashCommand = "./q_agent/run_simulation_Q"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd="../")
        _, _ = process.communicate()

        # Run the detection
        run_detection(brs=brs)

        # Get the reward
        with open("./feedback.pkl", "rb") as handle:
            feedback = pickle.load(handle)

        if len(feedback) > 0:
            for idx, row in features.iterrows():
                alert_type = row["type"]
                reward_obt = feedback[alert_type][0]
                reward_tot = feedback[alert_type][1]
                reward = reward_obt
        else:
            reward = 0

        # When the agent was caught, the game ends
        if reward == 0 and self.action_space[action][0] != 0:
            done = True
            if self.model == "SVM":
                reward = -1
            else:
                reward = -10000

        return next_state, reward, done
