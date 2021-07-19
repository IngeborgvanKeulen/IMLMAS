import json
import logging
import os
import random
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from mpire import WorkerPool
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tqdm import tqdm

from toy_examples.create_action_space import create_action_space

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BenchmarkEnvironment:
    def __init__(self, action_space: np.array, div_amount: int, div_account: int, div_days: int,
                 div_reward: int, brs: Dict, max_days: int, type: "str", **kwargs):
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
        reward = 0

        # Get the next state that is obtained by doing action in the current state
        next_state = np.copy(state)
        next_state[0] = np.copy(self.action_space[action][0])
        next_state[1] = np.copy(self.action_space[action][1])
        next_state[2] += int(np.copy(self.action_space[action][2]) * self.div_days) / self.max_days

        # Perform the AML detection method
        if self.actions_and_results[action][-1] == 0:
            done = True
        else:
            if self.type == "scatter-gather":
                reward = self.action_space[action][0] * self.div_amount *\
                         ((self.action_space[action][1] * self.div_account) - 2)
            else:
                reward = self.action_space[action][0] * self.div_amount * \
                         self.action_space[action][1] * self.div_account

        return next_state, reward, done


class DQN:
    """
    Deep Q Neural Network
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int,
                 loss: str, learning_rate: float, double: bool, verbose=0):
        """
        Initialize the Deep Q Neural Network

        :param state_dim: the number of features in a state, defines the shape of the input
        :param action_dim: the number of of possible actions, defines the shape of the output
        :param hidden_dim: the number of nodes in a hidden layer
        :param learning_rate: the learning rate
        :param double: whether two models, one for training and one for predicting, should be used
        :param verbose: whether training output should be shown in the console
        """
        # todo: optimistic weight initialization?
        # Build and compile the model
        self.model = keras.Sequential([
            Dense(hidden_dim, activation=tf.nn.leaky_relu, input_shape=[state_dim],
                  kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=1)),
            Dense(hidden_dim * 2, activation=tf.nn.leaky_relu,
                  kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=1)),
            Dense(action_dim, activation=keras.activations.linear,
                  kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=1)),
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=['mae', 'mse'])

        # Set whether the model should show output during training
        self.verbose = verbose

        self.double = double
        if self.double:
            # When using two models, then the target model should be a copy of the normal model
            self.target = keras.Sequential([
                Dense(hidden_dim, activation=tf.nn.leaky_relu, input_shape=[state_dim],
                      kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=1)),
                Dense(hidden_dim * 2, activation=tf.nn.leaky_relu,
                      kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=1)),
                Dense(action_dim, activation=keras.activations.linear,
                      kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=1)),
            ])
            self.target.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                loss=loss,
                                metrics=['mae', 'mse'])
            self.target_update()

    def update(self, states: np.array, targets: np.array, epochs: int):
        """
        Update the weights of the DQN based on the given training samples.

        :param states: the different states given as input to the network
        :param targets: the corresponding targets given as output to the network
        :param epochs: the number of epochs the network should train
        """
        # When only one sample is given, it should be reshaped
        states = np.array(states)
        if len(states.shape) == 1:
            states = states.reshape(1, -1)

        self.model.fit(np.array(states), np.array(targets), epochs=epochs, verbose=self.verbose)

    def target_update(self):
        """
        Copy the weights from the training model to the target model
        """
        self.target.set_weights(self.model.get_weights())

    def predict(self, state: np.array, use_target: bool = False) -> np.array:
        """
        Predict with the DQN the q-values for each action for the given state

        :param state: a state for which the q-values should be predicted
        :param use_target: whether to use the target model
        :return: the q-values corresponding to all actions for the given state
        """
        # When only one sample is given, it should be reshaped
        state = np.array(state)
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # Return the prediction (q-values) for the given state
        if self.double and use_target:
            return self.target.predict(np.array(state))
        else:
            return self.model.predict(np.array(state))

    def replay(self, memory: List, size: int, epochs: int, gamma: float,
               finite: int, sample_method: str, prio_buffer: bool) -> List:
        """
        Train on previous collected samples

        :param memory: collection of all encountered action-state pairs to train on
        :param size: the batch size the DQN should train on
        :param epochs: the number of epochs the DQN should train
        :param gamma: the discount factor used for the update function
        :param finite: the number of pairs in the memory buffer
        :param sample_method: which method to sample from the buffer
        (either random or max_rewards, where higher rewards pairs have more weight)
        :param prio_buffer: whether to prioritize certain samples to stay in the buffer
        :return: possibly updated memory buffer
        """
        if len(memory) >= size:
            # First update the buffer by resizing it to the given size. How to decide on which samples stays depends on
            # whether prio_buffer is True (keep highest rewards) or False (keep newer samples)
            if prio_buffer:
                finite = len(memory) if len(memory) <= finite else finite
                memory = sorted(memory, key=lambda x: x[3], reverse=True)[:finite]
            else:
                finite = 0 if len(memory) <= finite else finite
                memory = memory[-finite:]

            # Next sample data to train on (either weighted or random)
            if sample_method == "max_rewards":
                batch = random.choices(population=memory, weights=[x[3] for x in memory], k=size)
            else:
                batch = random.sample(memory, size)

            # Extract information from the data
            states = [x[0] for x in batch]
            actions = [x[1] for x in batch]
            next_states = [x[2] for x in batch]
            rewards = [x[3] for x in batch]
            not_dones = [float(not x[4]) for x in batch]

            # Predict the q-values for the next states; calculate the rewards
            q_values_next = self.predict(next_states, use_target=True)
            rewards = rewards + gamma * np.max(q_values_next, axis=1) * not_dones

            # Predict the q-values for the current states; set the actual targets
            q_values = self.predict(state=states, use_target=False)
            q_values[np.arange(0, len(q_values)), actions] = rewards

            self.update(states=states, targets=q_values, epochs=epochs)
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


def run_q_learning(env: BenchmarkEnvironment, model: DQN, params: Dict, show_output: bool) \
        -> Tuple[List, List, pd.DataFrame]:
    """
    Use q-learning in the given environment.

    :param env: the simulation environment
    :param model: the DQN model
    :param params: the parameters for the simulation
    :param show_output: whether to show the plots during learning
    :return: the final rewards, days without being caught per episode and all taken actions
    """
    # Set the most used parameters
    epochs = params["epochs"]
    max_days = params["max_days"]

    # Initialize the memory buffer; rewards; days per episode; dataframe for all actions
    memory = []
    final = []
    days_uncaught = []
    df = pd.DataFrame([], columns=["episode", "eps_episode", "day", "state", "action",
                                   "random", "reward", "next_state", "total"])
    stretched_eps = 0

    # Run for the given number of episodes
    for episode in tqdm(range(params["episodes"])):
        # Reset the environment
        state = env.reset()
        total = 0

        # Only start decreasing epsilon when learning has started
        if len(memory) < params["replay_size"]:
            stretched_eps += 1

        if params["stretched"]:
            ep = episode - stretched_eps
            # Obtain epsilon for this episode
            if params["eps_decay"] == "linear":
                eps_episode = max((0.01 - 1) * ep / (0.9 * params["episodes"]) + 1, 0.01)
            elif params["eps_decay"] == "annealing":
                eps_episode = max(pow(params["epsilon_factor"], ep), 0.01)
            else:
                eps_episode = params["epsilon"]

        else:
            # Obtain epsilon for this episode
            if params["eps_decay"] == "linear":
                eps_episode = max((0.01 - 1) * episode / (0.9 * params["episodes"]) + 1, 0.01)
            elif params["eps_decay"] == "annealing":
                eps_episode = max(pow(params["epsilon_factor"], episode), 0.01)
            else:
                eps_episode = params["epsilon"]

        # Update the target model when using a separate one and n_update steps have passed
        if model.double and episode % params["n_update"] == 0:
            model.target_update()

        day = 0
        # Run for the maximum number of days
        while day < max_days:

            # Either explore or exploit
            if random.random() < eps_episode or len(memory) < params["replay_size"]:
                # The taken action is random
                is_random = True

                # Select completely random an valid action
                valid = int(state[-1] * env.max_days) + env.action_space[:, -1] * env.div_days
                options = np.arange(len(env.action_space))[valid <= max_days]
                if len(options) == 0:
                    print("No options available (should be impossible)")
                    break
                action = random.choice(options)

            else:
                # The taken action is not random
                is_random = False

                # Find the best action
                q_values = model.predict(state=state, use_target=False)
                action = np.argmax(q_values)
                # In case this action is invalid, find the next-best action that is valid
                while int(state[-1] * env.max_days) + int(env.action_space[action][-1] * env.div_days) > max_days:
                    q_values[0][action] = -1
                    action = np.argmax(q_values)
                    if max(q_values[0]) == -1:
                        break

            # Perform the chosen action
            next_state, reward, done = env.step(state=state, action=action)

            # When the current day is the max number of days (or greater than), then the game is over
            day = int(next_state[-1] * env.max_days)
            if day >= max_days:
                done = True
                if day > max_days:
                    reward = 0

            # Update the total rewards; add experience to the memory; obtain q_values for the current state
            total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.predict(state=state, use_target=False)

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

            if done:
                # The run is done, so optionally train on the last sample and then break
                if not params["replay"]:
                    q_values[0][action] = reward
                    model.update(states=state, targets=q_values, epochs=epochs)
                else:
                    memory = model.replay(memory=memory, size=params["replay_size"], gamma=params["gamma"],
                                          epochs=epochs, finite=params["finite"], sample_method=params["sampling"],
                                          prio_buffer=params["prio_buffering"])
                break

            if params["replay"]:
                # Train the model by replaying a batch of the memory
                memory = model.replay(memory=memory, size=params["replay_size"], gamma=params["gamma"],
                                      epochs=epochs, finite=params["finite"], sample_method=params["sampling"],
                                      prio_buffer=params["prio_buffering"])
            else:
                # Train the model by using the last sample only
                q_values_next = model.predict(state=next_state, use_target=True)
                q_values[0][action] = reward + params["gamma"] * np.max(q_values_next)
                model.update(states=state, targets=q_values, epochs=epochs)

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
    model = DQN(action_dim=env.action_space.shape[0], state_dim=env.action_space.shape[1], **params["model"], verbose=0)

    # Run the simulation
    final, days, df = run_q_learning(env=env, model=model, params=params["simulation"], show_output=False)

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
            "loss": "mse",
            "learning_rate": 0.05,
            "hidden_dim": 128,
            "double": True,
        },
        "environment": {
            "div_amount": 7000,
            "div_account": 15,
            "div_days": 3,
            "div_reward": 1,
            "brs": {
                1: {
                    "use_br": True,
                    "threshold": 5000
                },
                2: {
                    "use_br": False,
                    "threshold": 5
                },
                3: {
                    "use_br": False,
                    "threshold": 30000
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
            "episodes": 1000,
            "sampling": "max_rewards",
            "prio_buffering": True,
            "finite": 500,
            "replay": True,
            "replay_size": 100,
            "n_update": 10,
            "stretched": False,
            "epsilon": .1,
            "epsilon_factor": 0.998,
            "eps_decay": "linear",
            "epochs": 150,
            "gamma": 0.99,
            "goal": 5000 * 3 * (15-2),
            "max_days": 3,
            "title": "Launder as much as possible",
        },
        "file_path": "./results/thesis/dqn/",
    }
    run_experiment(params=parameters)


def parameter_sweep(args: Namespace):
    """
    This function runs the simulation for all different parameter settings in todo_params.
    At the end all simulation are evaluated by looking at the last 50 games.

    :param args: the jobs and cpu ids arguments
    """
    todo_params = []
    for epi in [1000, 1500]:
        for size in [100]:
            for update in [10]:
                for epochs in [150]:
                    for repeat in [1, 2]:
                        todo_params.append([{
                            "model": {
                                "loss": "mse",
                                "learning_rate": 0.05,
                                "hidden_dim": 128,
                                "double": True,
                            },
                            "environment": {
                                "div_amount": 7000,
                                "div_account": 15,
                                "div_days": 3,
                                "div_reward": 1,
                                "brs": {
                                    1: {
                                        "use_br": True,
                                        "threshold": 5000
                                    },
                                    2: {
                                        "use_br": False,
                                        "threshold": 5
                                    },
                                    3: {
                                        "use_br": False,
                                        "threshold": 30000
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
                                "episodes": epi,
                                "sampling": "max_rewards",
                                "prio_buffering": True,
                                "finite": 500,
                                "replay": True,
                                "replay_size": size,
                                "n_update": update,
                                "stretched": False,
                                "epsilon": .1,
                                "epsilon_factor": 0.998,
                                "eps_decay": "linear",
                                "epochs": epochs,
                                "gamma": 0.99,
                                "goal": 5000 * 3 * (15-2),
                                "max_days": 3,
                                "title": "Launder as much as possible",
                            },
                            "file_path": "./results/thesis/dqn/",
                        }])

    # Run the settings in parallel
    with WorkerPool(cpu_ids=args.cpu_ids, n_jobs=args.n_jobs) as wp:
        wp.map(run_experiment, todo_params, progress_bar=True)

    # Evaluate all results
    parameters = todo_params[-1][0]
    dirs = os.listdir(parameters["file_path"])
    df = pd.DataFrame([])
    for file_dir in dirs:
        if file_dir.endswith("csv"):
            continue
        # Get the parameters for this specific simulation
        params = pd.read_json(os.path.join(parameters["file_path"], file_dir, "params.json"), orient="index")
        new_df = params.loc["simulation"]
        new_df = new_df.append(pd.Series([file_dir], index=["file"]))
        new_df = new_df.dropna()

        # Get the rewards obtained per game and calculate the mean and std for the last 50 games
        results = pd.read_csv(os.path.join(parameters["file_path"], file_dir, "action_states.csv"))
        total = results.drop_duplicates(subset=["episode"], keep="last")["total"]
        new_df["average"] = np.mean(total[-50:])
        new_df["deviation"] = np.std(total[-50:])
        new_df = pd.DataFrame(new_df).T
        new_df = new_df.set_index("file")

        # Combine the information in one single file
        df = pd.concat([df, new_df])
    df.to_csv(os.path.join(parameters["file_path"], "evaluation.csv"))


def parse_args():
    """
    Parse command line interface (CLI) arguments.

    :return: CLI arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--n-jobs",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--cpu-ids",
        nargs='+',
        default=[],
    )

    args = parser.parse_args()
    args.cpu_ids = [int(i) for i in args.cpu_ids]
    args.cpu_ids = [i for i in range(args.cpu_ids[0], args.cpu_ids[1] + 1)]
    return args


def main():
    args = parse_args()
    # single_run()
    parameter_sweep(args=args)


if __name__ == "__main__":
    main()
