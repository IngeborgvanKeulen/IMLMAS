import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from tqdm import tqdm

from toy_examples.create_action_space import create_action_space

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BenchmarkEnvironment:
    def __init__(self, action_space: np.array, div_amount: int, div_account: int, div_days: int,
                 div_reward: int, max_days: int, type: str, **kwargs):
        """
        Initialize the environment

        :param action_space: all the possible actions
        :param div_amount: the normalization value for the feature amount
        :param div_account: the normalization value for the feature account
        :param div_days: the normalization value for the feature days
        :param div_reward: the normalization value for the reward
        :param max_days: the max number of days the agent can try launder money
        """
        self.all_options = action_space
        self.action_space = action_space[:, :-1]
        self.actions_and_results = action_space
        self.div_amount = div_amount
        self.div_account = div_account
        self.div_days = div_days
        self.div_reward = div_reward
        self.max_days = max_days
        self.type = type

    def reset(self):
        """
        Reset the complete environment

        :return: the start state
        """
        return np.array([0 / self.div_amount, 0 / self.div_account, 0 / self.max_days])

    def step(self, state: np.array, action: int):
        """
        Perform the action and return the results of doing this action

        :param state: the current state
        :param action: the chosen action (index in the action space)
        :return: the next state the agent ended up in, the obtained reward and whether the agent is caught (done = True)
        """
        # Set the defaults
        done = False
        reward = -1

        # Get the next state that is obtained by doing action in the current state
        next_state = np.copy(state)
        next_state[0] = np.copy(self.action_space[action][0])
        next_state[1] = np.copy(self.action_space[action][1])
        next_state[2] = np.copy(self.action_space[action][2])

        # Perform the AML detection method
        if self.actions_and_results[action][-1] == 0:
            done = True
        else:
            if self.type == "scatter-gather":
                reward = self.action_space[action][0] * self.div_amount * \
                         ((self.action_space[action][1] * self.div_account) - 2)
            else:
                reward = self.action_space[action][0] * self.div_amount * \
                         self.action_space[action][1] * self.div_account

        return next_state, reward, done


class SVM:
    """
    Suppport Vector Machine
    """

    def __init__(self, kernel: str = "rbf", regularization: int = 1000, tol: float = 1e-3, gamma: str = "scale"):
        """
        Initialize the SVM

        :param kernel: the kernel the SVM should use
        :param regularization: the regularization parameter defining the importance of preventing misclassifications
        :param tol: the error tolerance before stopping with training
        :param gamma: the gamma value
        """
        # Set the model
        self.model = svm.SVC(kernel=kernel, C=regularization, tol=tol, gamma=gamma)

    def update(self, states: np.array, targets: np.array):
        """
        Update the weights of the SVM based on the given training samples.

        :param states: the different states given as input to the network
        :param targets: the corresponding targets given as output to the network
        """
        # When only one sample is given, it should be reshaped
        states = np.array(states)
        if len(states.shape) == 1:
            states = states.reshape(1, -1)

        self.model.fit(np.array(states), np.array(targets).astype("float"))

    def predict(self, action: np.array) -> np.array:
        """
        Predict with the SVM the success for each action

        :param action: an action (array) for which the success (1 or 0) should be predicted
        :return: the success prediction
        """
        # When only one sample is given, it should be reshaped
        action = np.array(action)
        if len(action.shape) == 1:
            action = action.reshape(1, -1)

        # Return the prediction for the given action
        return self.model.predict(np.array(action))

    def replay(self, memory: List, finite: int, prio_buffer: bool) -> List:
        """
        Train on previous collected samples

        :param memory: collection of all encountered action-state pairs to train on
        :param finite: the number of pairs in the memory buffer
        :param prio_buffer: whether to prioritize certain samples to stay in the buffer
        :return: possibly updated memory buffer
        """
        # First update the buffer by resizing it to the given size. How to decide on which samples stays depends on
        # whether prio_buffer is True (keep highest rewards) or False (keep newer samples)
        if prio_buffer:
            finite = len(memory) if len(memory) <= finite else finite
            memory = sorted(memory, key=lambda x: x[1], reverse=True)[:finite]
        else:
            finite = 0 if len(memory) <= finite else finite
            memory = memory[-finite:]

        # Extract information from the data
        states = [x[0] for x in memory]
        rewards = [x[1] for x in memory]

        if len(np.unique(rewards)) == 1:
            # When there is only one label in the data, postpone training until data set contains both labels
            return memory

        self.update(states=states, targets=rewards)

        return memory


def plot_and_save_results(rewards: List, days_uncaught: List, title: str,
                          goal: int, max_days: int, file_path: Optional[str] = None):
    """
    Plot the rewards obtained in each episode and the number of days the agent was not caught in an episode.

    :param rewards: the obtained rewards up to now in each episode
    :param days_uncaught: the number of days the agent was not being caught in an episode
    :param title: the title of the graph
    :param goal: the maximum possible obtainable goal
    :param max_days: the maximum number of days in an episode
    :param file_path: the file path where the graph should be saved
    """
    # Close all previous windows
    plt.close("all")

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(rewards, label="score per run")
    ax[0].axhline(goal, c="red", ls="--", label="maximum goal")
    ax[0].set_xlabel("Games")
    ax[0].set_ylabel("Total Reward")
    ax[0].legend()

    # Plot the days over episodes
    ax[1].plot(days_uncaught)
    ax[1].set_xlabel("Games")
    ax[1].set_ylabel("# days")
    ax[1].axhline(max_days, c="red", ls="--", label="maximum days")

    # If a file path is given, then the plot should be saved otherwise it should be shown
    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.show()


def run_learning(env: BenchmarkEnvironment, model: SVM, params: Dict, show_output: bool) \
        -> Tuple[List, List, pd.DataFrame]:
    """
    Use svm learning in the given environment.

    :param env: the simulation environment
    :param model: the svm model
    :param params: the parameters for the simulation
    :param show_output: whether to show the plots during learning
    :return: the final rewards, days without being caught per episode and all taken actions
    """
    # Set the most used parameters
    max_days = params["max_days"]

    # Initialize the memory buffer; rewards; days per episode; dataframe for all actions
    memory = []
    final = []
    days_uncaught = []
    df = pd.DataFrame([], columns=["episode", "eps_episode", "day", "state", "action",
                                   "random", "reward", "next_state", "total"])

    # Run for the given number of episodes
    for episode in tqdm(range(params["max_episodes"])):
        # Reset the environment
        state = env.reset()
        total = 0

        # Obtain epsilon for this episode
        if params["eps_decay"] == "linear":
            eps_episode = 0
            if params["e_episodes"] >= 1:
                eps_episode = max((0.01 - 1) * episode / params["e_episodes"] + 1, 0.01)
        elif params["eps_decay"] == "annealing":
            eps_episode = max(pow(params["epsilon_factor"], episode), 0.01)
        else:
            eps_episode = params["epsilon"]

        day = 0
        # Run for the maximum number of days
        while day < max_days:
            # Either explore or exploit
            if random.random() < eps_episode:
                # The taken action is random
                is_random = True

                # Select completely random an valid action
                valid = day + env.action_space[:, -1] * env.div_days
                options = np.arange(len(env.action_space))[valid <= max_days]
                if len(options) == 0:
                    print("No options available (should be impossible)")
                    break
                action = random.choice(options)

            else:
                # The taken action is not random
                is_random = False

                # Find the best action
                q_values = model.predict(action=env.action_space)
                exp_rew = env.action_space[:, 0] * env.div_amount * env.action_space[:, 1] * env.div_account \
                          / (env.action_space[:, 2] * env.div_days)
                action = np.argmax(q_values * exp_rew)

                # In case this action is invalid, find the next-best action that is valid
                while day + round(env.action_space[action][-1] * env.div_days) > max_days:
                    q_values[action] = -1
                    action = np.argmax(q_values * exp_rew)
                    if max(q_values) == -1:
                        break

            # Perform the chosen action
            next_state, reward, done = env.step(state=state, action=action)

            # When the current day is the max number of days (or greater than), then the game is over
            day += round(next_state[-1] * env.div_days)
            if day >= max_days:
                done = True
                if day > max_days:
                    reward = 0

            # Update the total rewards; add experience to the memory; obtain q_values for the current state
            total += 0 if reward == -1 else reward

            # Add the current action and context information to the dataframe
            df = df.append({"episode": episode,
                            "eps_episode": eps_episode,
                            "day": day,
                            "state": state * [env.div_amount, env.div_account, env.max_days],
                            "action": env.action_space[action] * [env.div_amount, env.div_account, env.div_days],
                            "random": is_random,
                            "reward": reward,
                            "next_state": next_state * [env.div_amount, env.div_account, env.max_days],
                            "total": total}, ignore_index=True)

            if reward != -1:
                reward = 1
            memory.append((next_state, reward))

            if params["replay"]:
                # Train the model by replaying a batch of the memory
                memory = model.replay(memory=memory, finite=params["finite"], prio_buffer=params["prio_buffering"])
            else:
                # Train the model by using the last sample only
                model.update(states=state, targets=[reward])

            if done:
                # The run is done, so optionally train on the last sample and then break
                break

            # Go to the next state
            state = next_state

        # Add the results and optionally plot them
        days_uncaught.append(day)
        final.append(total)
        if show_output:
            plot_and_save_results(rewards=final, days_uncaught=days_uncaught, title=params["title"],
                                  goal=params["goal"], max_days=params["max_days"])

    return final, days_uncaught, df


def store_results(params: Dict, final: List, days: List, df: pd.DataFrame):
    """
    Store the results from the complete simulation.

    :param params: the params used in the simulation
    :param final: the final rewards obtained in each episode
    :param days: the number of days of being uncaught in each episode
    :param df: a dataframe containing all actions taken with extra information like randomness
    """
    # Create a directory name with the current time, amount steps and optionally count when the dir already exists
    now = datetime.now()
    cnt = 0
    dir_name = now.strftime(f'setting_{params["action_space"]["step_amount"]}_{cnt}')

    # Make sure all paths exist
    if not os.path.exists(params["file_path"]):
        os.makedirs(params["file_path"])
    dir_path = os.path.join(params["file_path"], dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print("Create new folder")
        while os.path.exists(dir_path):
            cnt += 1
            dir_name = now.strftime(f'setting_{params["action_space"]["step_amount"]}_{cnt}')
            dir_path = os.path.join(params["file_path"], dir_name)
        os.makedirs(dir_path)

    # Save the parameters
    with open(os.path.join(dir_path, "params.json"), "w") as fp:
        params.pop("file_path", None)
        json.dump(params, fp, indent=4)

    # Create a plot and save it
    title = "you have {} days".format(params["simulation"]["max_days"])
    file_path = os.path.join(dir_path, "plot.png")
    plot_and_save_results(final, days, title, params["simulation"]["goal"], params["simulation"]["max_days"], file_path)

    # Save the dataframe
    df.to_csv(os.path.join(dir_path, "action_states.csv"))


def run_experiment(params: Dict[str, Dict]):
    """
    This function runs the experiments for the given parameters and stores the results at the end.

    :param params: a dictionary containing all different parameters for all different modules.
    """
    # Create the action space; initialize the environment; initialize the DQN model
    action_space = create_action_space(params=params["action_space"], **params["environment"])
    env = BenchmarkEnvironment(action_space=action_space, **params["environment"],
                               max_days=params["simulation"]["max_days"], type=params["simulation"]["type"])
    model = SVM(**params["model"])

    # Run the simulation
    final, days, df = run_learning(env=env, model=model, params=params["simulation"], show_output=False)

    # Update the parameters with the number of features
    params["model"].update({"number of features": env.action_space.shape[1]})
    params["model"].update({"business rules": ["amount < 5000"]})

    # Store the results
    store_results(params, final, days, df)


def single_run():
    """
    This function runs the simulation for one parameter setting.
    """
    parameters = {
        "model": {
            "kernel": "rbf",
            "regularization": 1000,
            "tol": 1e-3,
            "gamma": "scale",
        },
        "environment": {
            "div_amount": 7000,
            "div_account": 15,
            "div_days": 3,
            "div_reward": (15 - 2) * 7000,
            "brs": {
                1: {
                    "use_br": True,
                    "threshold": 5000
                },
                2: {
                    "use_br": False,
                    "threshold": 7
                },
                3: {
                    "use_br": False,
                    "threshold": 22000
                }
            }
        },
        "action_space": {
            "success": 1,
            "caught": 0,
            "min_amount": 0,
            "step_amount": 1000,
            "min_account": 4,
            "step_account": 1,
            "min_days": 1,
            "step_days": 1,
        },
        "simulation": {
            "type": "scatter-gather",
            "e_episodes": 1,
            "max_episodes": 1,
            "prio_buffering": True,
            "finite": 30000,
            "replay": True,
            "epsilon": .1,
            "epsilon_factor": 0.998,
            "eps_decay": "annealing",
            "goal": 5000 * 3 * (15 - 2),
            "max_days": 3,
            "title": "Launder as much as possible",
        },
        "file_path": "./results/test_toy_svm/",
    }
    run_experiment(params=parameters)


def main():
    single_run()


if __name__ == "__main__":
    main()
