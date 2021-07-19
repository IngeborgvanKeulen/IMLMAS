import json
import logging
import os
import random
from argparse import ArgumentParser
from datetime import datetime
from typing import List

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
    def __init__(self, action_space, div_amount, div_account, div_days, div_reward, brs, max_days, type):
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
        return np.array([0 / self.div_amount, 0 / self.max_days])

    def step(self, state, action):
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
        # todo: 07/07 test this
        next_state[1] += self.action_space[action][2]
        # next_state[1] += round(np.copy(self.action_space[action][2]) * self.div_days) / self.max_days

        # todo: make this less hardcoded
        # Perform the AML detection method
        if self.actions_and_results[action][-1] == 0:
            done = True
            reward = -10000
        else:
            reward = self.action_space[action][0] * self.div_amount * \
                     ((self.action_space[action][1] * self.div_account) - 2)

        return next_state, reward, done


class BootstrapDQN:
    """
    Bootstrapped Deep Q Neural Network
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, loss: str,
                 learning_rate: float, double: bool, n_heads: int, verbose=0, **kwargs):
        """
        Initialize the Deep Q Neural Network

        :param state_dim: the number of features in a state, defines the shape of the input
        :param action_dim: the number of of possible actions, defines the shape of the output
        :param hidden_dim: the number of nodes in a hidden layer
        :param learning_rate: the learning rate
        :param double: whether two models, one for training and one for predicting, should be used
        :param verbose: whether training output should be shown in the console
        """
        reload_path = None
        # reload_path = "./head_0.h5"
        # reload_path = "/home/ingeborg.local/Thesis/FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_2/3_days/episodes_1500_heads_10_prob_0.5/setting_1000_14/head_0.h5"
        # todo: need seed?
        self.random_state = np.random.RandomState()

        # Build and compile the model
        model = keras.Sequential([
            Dense(hidden_dim, activation=tf.nn.leaky_relu, input_shape=[state_dim],
                  kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
            Dense(hidden_dim * 2, activation=tf.nn.leaky_relu,
                  kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
        ])

        self.heads = []
        for i in range(n_heads):
            self.heads.append(tf.keras.Sequential(
                [model,
                 Dense(action_dim, activation=keras.activations.linear,
                       kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10))]))
            self.heads[i].compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                  loss=loss,
                                  metrics=['mae', 'mse'])

        if reload_path is not None:
            new_model = tf.keras.models.load_model(reload_path, custom_objects={"leaky_relu": tf.nn.leaky_relu})
            new_weights = new_model.get_weights()[:4] + self.heads[0].get_weights()[4:]
            self.heads[0].set_weights(new_weights)

        # Set whether the model should show output during training
        self.verbose = verbose

        self.double = double
        if double:
            # When using two models, then the target model should be a copy of the normal model
            target = keras.Sequential([
                Dense(hidden_dim, activation=tf.nn.leaky_relu, input_shape=[state_dim],
                      kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
                Dense(hidden_dim * 2, activation=tf.nn.leaky_relu,
                      kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
            ])
            self.target_heads = []
            for i in range(n_heads):
                self.target_heads.append(tf.keras.Sequential(
                    [target, Dense(action_dim, activation=keras.activations.linear)]))
                self.target_heads[i].compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                             loss=loss,
                                             metrics=['mae', 'mse'])
                self.target_update(idx=i)

    def store_single_head(self, path, epoch):
        model = self.heads[0]
        weights = model.get_weights()
        with open(os.path.join(path, f"head_0_{epoch}.npy"), 'wb') as f:
            np.save(f, weights)

    def store_heads(self, path):
        """
        Store the training models to the given path
        :param path: path where models should be stored
        """
        for idx, model in enumerate(self.heads):
            model.save(os.path.join(path, f"head_{idx}.h5"))

        ## It can be used to reconstruct the model identically
        # reconstructed_model = keras.models.load_model("my_h5_model.h5")


    def update(self, states: tf.Tensor, targets: tf.Tensor, epochs: int, idx: int):
        """
        Update the weights of the DQN based on the given training samples.

        :param states: the different states given as input to the network
        :param targets: the corresponding targets given as output to the network
        :param epochs: the number of epochs the network should train
        :param idx: which head to use to predict
        """
        self.heads[idx].fit(states, targets, epochs=epochs, verbose=self.verbose)

    def target_update(self, idx: int):
        """
        Copy the weights from the training model to the target model
        :param idx: which head to use to predict
        """
        self.target_heads[idx].set_weights(self.heads[idx].get_weights())

    def predict(self, state: tf.Tensor, idx: int, use_target: bool = False):
        """
        Predict with the DQN the q-values for each action for the given state

        :param state: a state for which the q-values should be predicted
        :param idx: which head to use to predict
        :param use_target: whether to use the target model
        :return: the q-values corresponding to all actions for the given state
        """
        if self.double and use_target:
            return self.target_heads[idx].predict_step(state)
        else:
            return self.heads[idx].predict_step(state)

    def replay(self, memory: List, size: int, epochs: int, gamma: float, finite: int,
               sample_method: str, prio_buffer: bool) -> List:
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
        :return: the possible changed memory buffer
        """
        if len(memory) >= size:
            if prio_buffer:
                finite = len(memory) if len(memory) <= finite else finite
                memory = sorted(memory, key=lambda x: x[3], reverse=True)[:finite]
            else:
                finite = 0 if len(memory) <= finite else finite
                # Randomly sample size samples from the memory
                memory = memory[-finite:]

            if sample_method == "max_rewards":
                batch = random.choices(population=memory, weights=[x[3] for x in memory], k=size)
            else:
                batch = random.sample(memory, size)

            # Extract information from the data
            states = np.array([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            next_states = np.array([x[2] for x in batch])
            rewards = np.array([x[3] for x in batch])
            not_dones = np.array([float(not x[4]) for x in batch])
            masks = np.array([x[5] for x in batch])
            days = [x[6] for x in batch]
            next_days = [x[7] for x in batch]

            next_states[:, 0] = next_states[:, 0] / (np.array(next_days) * 7000 * 13)
            states[:, 0] = states[:, 0] / (7000 * 13 * np.array(days))

            def train_each_head(head_idx):
                head_idx = head_idx[0]
                mask = masks[:, head_idx]
                if sum(mask) == 0:
                    return
                next_states_c = tf.convert_to_tensor(next_states[mask == 1])
                states_c = tf.convert_to_tensor(states[mask == 1])

                # Predict the q-values for the next states; calculate the rewards
                q_values_next = self.predict(next_states_c, idx=head_idx, use_target=True)
                rewards_t = rewards[mask == 1] + gamma * np.max(q_values_next, axis=1).T * not_dones[mask == 1]

                # Predict the q-values for the current states; set the actual targets
                q_values = self.predict(state=states_c, use_target=False, idx=head_idx).numpy()
                q_values[np.arange(0, len(q_values)), actions[mask == 1]] = rewards_t

                # print("-----------------------------")
                # print(states_c)
                # print("-----------------------------")
                # print(q_values_next)
                # print(rewards_t)
                # print("-----------------------------")
                # if None in q_values:
                #     raise ValueError("Nan found")
                # if None in rewards_t:
                #     raise ValueError("Nan found")
                self.update(states=states_c, targets=q_values, epochs=epochs, idx=head_idx)

            head_list = list(np.arange(len(self.heads)))
            random.shuffle(head_list)
            for i in head_list:
                train_each_head([i])

        return memory


def plot_and_save_results(rewards, days_uncaught, title, goal, max_days, file_path=None):
    """
    Plot the rewards obtained in each episode and the number of days the agent was not caught in an episode

    :param rewards: the obtained rewards up to now in each episode
    :param days_uncaught: the number of days the agent was not being caught in an episode
    :param title: the title of the graph
    :param goal: the maximum possible obtainable goal
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


def run_q_learning(env, model, params, show_output):
    """
    Use an q-learning in the given environment

    :param env: the simulation environment
    :param model: the DQN model
    :param params: the parameters for the simulation
    :param show_output: whether to show the plots during learning
    :return: the final rewards and days without being caught per episode
    """
    df = pd.DataFrame([],
                      columns=["episode", "head", "eps_episode", "day", "state", "action", "random",
                               "reward", "next_state", "total"])
    # Set the most used parameters
    epochs = params["epochs"]

    # Initialize the memory buffer, rewards and days per episode
    memory = []
    final = []
    days_uncaught = []
    stretched_eps = 0

    # Run for the given number of episodes
    for episode in tqdm(range(params["max_episodes"])):

        # Reset state
        state = env.reset()
        total = 0
        head_idx = random.randint(0, len(model.heads) - 1)

        # Only start decreasing epsilon when learning has started
        if len(memory) < params["replay_size"]:
            stretched_eps += 1

        if params["stretched"]:
            ep = episode - stretched_eps
            # Obtain epsilon for this episode
            if params["eps_decay"] == "linear":
                eps_episode = max((0.01 - 1) * ep / params["e_episodes"] + 1, 0.01)
            elif params["eps_decay"] == "annealing":
                eps_episode = max(pow(params["epsilon_factor"], ep), 0.01)
            else:
                eps_episode = params["epsilon"]

        else:
            # Obtain epsilon for this episode
            if params["eps_decay"] == "linear":
                eps_episode = 0
                if params["e_episodes"] >= 1:
                    eps_episode = max((0.01 - 1) * episode / params["e_episodes"] + 1, 0.01)
            elif params["eps_decay"] == "annealing":
                eps_episode = max(pow(params["epsilon_factor"], episode), 0.01)
            else:
                eps_episode = params["epsilon"]

        # Update the target model when using a separate one and n_steps have passed
        if model.double and episode % params["n_update"] == 0:
            for i in range(len(model.heads)):
                model.target_update(idx=i)

        day = 0
        # model.store_single_head(path="temporary_single_head/", epoch=episode)
        # Run for the maximum number of days
        while day < params["max_days"]:
            better_action = True
            is_random = False
            # Use a greedy search policy
            if random.random() < eps_episode or len(memory) < params["replay_size"]:
                is_random = True

                # Select completely random
                valid = round(state[-1] * env.max_days) + env.action_space[:, -1] * env.div_days
                options = np.arange(len(env.action_space))[valid <= params["max_days"]]
                if len(options) == 0:
                    raise ValueError("No options available (should be impossible)")
                action = random.choice(options)
            else:
                # Find the best action
                norm_state = state.copy()
                if day != 0:
                    # todo: how to do norm
                    norm_state = norm_state / [env.div_amount * env.div_account * day, 1]
                norm_state = tf.convert_to_tensor(norm_state.reshape(1, -1))
                q_values = model.predict(state=norm_state, idx=head_idx, use_target=False).numpy()

                # First find the best action
                action = np.argmax(q_values)
                while round(state[-1] * env.max_days) + round(env.action_space[action][-1] * env.div_days) \
                        > params["max_days"]:
                    q_values[0][action] = -1
                    action = np.argmax(q_values)
                    if max(q_values[0]) == -1:
                        better_action = False
                        break

            # Perform the chosen action; if this is the last day then done should be set to True
            next_state, reward, done = env.step(state=state, action=action)

            if round(next_state[-1] * env.max_days) == day:
                raise ValueError("Day did not change")
            day = round(next_state[-1] * env.max_days)
            if day == params["max_days"]:
                done = True
            elif day > params["max_days"]:
                # print(df)
                # print(next_state)
                # print(state)
                # print(day)
                if better_action:
                    raise ValueError("should not be true")
                done = True
                reward = 0

            # Update the total rewards; add experience to the memory; obtain q_values for the current state
            total += reward if reward > 0 else 0
            next_state[0] = total
            cur_day = round(state[-1] * env.max_days)

            memory.append((state, action, next_state, reward, done,
                           model.random_state.binomial(1, params["probability"], len(model.heads)),
                           cur_day if cur_day != 0 else 1, day))
            if not params["replay"]:
                q_values = model.predict(
                    state=state / [env.div_amount * env.div_account * round(state[-1] * env.max_days), 1], idx=head_idx,
                    use_target=False)

            # Add the current action and context information to the dataframe
            df = df.append({"episode": episode,
                            "head": head_idx,
                            "eps_episode": eps_episode,
                            "day": day,
                            "state": state,
                            "action": env.action_space[action] * [env.div_amount, env.div_account, env.div_days],
                            "random": is_random,
                            "reward": reward,
                            "next_state": next_state,
                            "total": total}, ignore_index=True)

            if len(df) > 1:
                if reward < 0:
                    if df["action"].iloc[-1][0] == 0:
                        print(df)
                        raise ValueError("AMOUNT IS ZERO BUT REWARD NEGATIVE")
            # if len(df) > 50 and sum(df["reward"][-30:]) == 0 and len(set(df["reward"][-30:].values)) == 1:
            #     print(df)
            #     # model.store_heads(path="./temporary/")
            #     raise ValueError("always zero")


            if done:
                # The run is done, so optionally train on the last sample and then break
                if not params["replay"]:
                    q_values[0][action] = reward
                    model.update(states=state, targets=q_values, epochs=epochs, idx=head_idx)
                else:
                    memory = model.replay(memory=memory, size=params["replay_size"], gamma=params["gamma"],
                                          epochs=epochs, finite=params["finite"], sample_method=params["sampling"],
                                          prio_buffer=params["prio_buffering"])
                break

            if params["replay"]:
                # Train the model by replaying a batch of the memory
                # todo: train every model or only current one?
                memory = model.replay(memory=memory, size=params["replay_size"], gamma=params["gamma"], epochs=epochs,
                                      finite=params["finite"], sample_method=params["sampling"],
                                      prio_buffer=params["prio_buffering"])
            else:
                # Train the model by using the last sample only
                q_values_next = model.predict(state=next_state, idx=head_idx, use_target=True)
                q_values[0][action] = reward + params["gamma"] * np.max(q_values_next)
                model.update(states=state, targets=q_values, epochs=epochs, idx=head_idx)

            # Go to the next state
            state = next_state

        # Add the results and plot them
        days_uncaught.append(day)
        final.append(total)
        if show_output:
            plot_and_save_results(rewards=final, days_uncaught=days_uncaught, title=params["title"],
                                  goal=params["goal"], max_days=params["max_days"])

    return final, days_uncaught, df


def store_results(params, final, days, df, model):
    """
    Store the results from the complete simulation

    :param params: the params used in the simulation
    :param final: the final rewards obtained in each episode
    :param days: the number of days of being uncaught in each episode
    :param model: the model
    """
    # Save all parameters and the plot in one directory
    now = datetime.now()
    cnt = 0
    tmp = params["action_space"]["step_amount"]
    dir_name = now.strftime(f"setting_{tmp}_{cnt}")

    if not os.path.exists(params["file_path"]):
        os.makedirs(params["file_path"])

    dir_path = os.path.join(params["file_path"], dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print("Create new folder")
        while os.path.exists(dir_path):
            cnt += 1
            # dir_name = now.strftime(f"%d_%m_%Y-%H_%M_%S_{cnt}")
            dir_name = now.strftime(f"setting_{tmp}_{cnt}")
            dir_path = os.path.join(params["file_path"], dir_name)
        os.makedirs(dir_path)

    with open(os.path.join(dir_path, "params.json"), "w") as fp:
        params.pop("file_path", None)
        json.dump(params, fp, indent=4)

    # Create plot and save it
    title = "you have {} days".format(params["simulation"]["max_days"])
    file_path = os.path.join(dir_path, "plot.png")
    plot_and_save_results(final, days, title, params["simulation"]["goal"], params["simulation"]["max_days"], file_path)

    # Save dataframe
    df.to_csv(os.path.join(dir_path, "action_states.csv"))

    # Save models
    model.store_heads(path=dir_path)


def run_experiment(params):
    # Initialize the environment and DQN model
    action_space = create_action_space(params=params["action_space"], **params["environment"])
    env = BenchmarkEnvironment(action_space=action_space, **params["environment"],
                               max_days=params["simulation"]["max_days"], type=params["simulation"]["type"])
    model = BootstrapDQN(action_dim=env.action_space.shape[0], state_dim=2, **params["model"],
                         verbose=0)

    # Run the simulation
    final, days, df = run_q_learning(env=env, model=model, params=params["simulation"], show_output=False)

    # Update the params
    params["model"].update({"number of features": env.action_space.shape[1]})

    # Store the results
    store_results(params, final, days, df, model)


def single_run():
    # Set the parameters (this is the only thing you should touch)
    parameters = {
        "model": {
            "loss": "mse",
            "learning_rate": 0.05,
            "hidden_dim": 128,
            "double": True,
            "n_heads": 10,
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
            "max_episodes": epi,
            "e_episodes": epi,
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
            "probability": 0.5,
            "converge_steps": -10 * 10,
            "gamma": 0.99,
            "goal": 5000 * 3 * (15-2),
            "max_days": 3,
            "title": "Launder as much as possible",
        },
        "file_path": "./results/test_run/",
    }
    run_experiment(params=parameters)


def parameter_sweep(args):
    todo_params = []
    # for n_days, lr in [(3, 0.01), (3, 0.05), (3, 0.075)]:
    for n_days, lr in [(5, 0.05)]:
        for decay_type in ["linear"]:
            for sampling_type, e_epi, max_epi in [("max_rewards", 1000, 2500)]:
                for use_first, use_second, use_third in [(False, True, False)]:
                    for prio_buff in [False]:
                        todo_params.append([{
                            "model": {
                                "loss": "mse",
                                "learning_rate": lr,
                                "hidden_dim": 128,
                                "double": True,
                                "n_heads": 10,
                            },
                            "environment": {
                                "div_amount": 7000,
                                "div_account": 15,
                                "div_days": n_days,
                                "div_reward": 1,
                                "brs": {
                                    1: {
                                        "use_br": use_first,
                                        "threshold": 5000
                                    },
                                    2: {
                                        "use_br": use_second,
                                        "threshold": 7
                                    },
                                    3: {
                                        "use_br": use_third,
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
                                # DQN
                                "epochs": 150,
                                "prio_buffering": prio_buff,
                                "finite": 500,
                                "replay": True,
                                "replay_size": 75,
                                "sampling": sampling_type,
                                "n_update": 50,
                                "gamma": 0.99,

                                # Bootstrap
                                "probability": 0.5,
                                "converge_steps": -5 * 10,

                                # Exploration
                                "epsilon": .1,
                                "epsilon_factor": 0.998,
                                "eps_decay": decay_type,
                                "stretched": False,

                                # General
                                "max_days": n_days,
                                "e_episodes": e_epi,
                                "max_episodes": max_epi,
                                "type": "scatter-gather",

                                # Plots
                                "goal": 80000,
                                "title": "Launder as much as possible",
                            },
                            "file_path": f"../../FINAL_RESULTS_UPDATED/Experiments/Experiment_final/BDQN/business_rule_{2 if use_second else 3 if use_third else 12}/{n_days}_days/normal/learning_rate_0.05/{decay_type}_{sampling_type}_{e_epi}_{max_epi}/"
                        }])

    with WorkerPool(cpu_ids=args.cpu_ids, n_jobs=args.n_jobs) as wp:
        wp.map(run_experiment, todo_params, progress_bar=True)

    # # Evaluate all results
    # parameters = todo_params[-1][0]
    # # parameters["file_path"] = "../goliath_results/AMAP/bootstrap_conv_06_04_2021"
    # # parameters["file_path"] = "../goliath_results/AMAP/bootstrap_conv_09_04_2021"
    # dirs = os.listdir(parameters["file_path"])
    # df = pd.DataFrame([])
    # for file_dir in dirs:
    #     if file_dir.endswith("evaluation.csv"):
    #         continue
    #
    #     params = pd.read_json(os.path.join(parameters["file_path"], file_dir, "params.json"), orient="index")
    #     new_df = params.loc["simulation"]
    #     new_df["number of heads"] = params.loc["model"]["n_heads"]
    #     new_df = new_df.append(pd.Series([file_dir], index=["file"]))
    #     new_df = new_df.dropna()
    #     results = pd.read_csv(os.path.join(parameters["file_path"], file_dir, "action_states.csv"))
    #     if "episode" in results.columns:
    #         total = results.drop_duplicates(subset=["episode"], keep="last")["total"]
    #         new_df["average"] = np.mean(total[-50:])
    #         new_df["deviation"] = np.std(total[-50:])
    #     else:
    #         indices = np.arange(len(results))[results["day"] == 1]
    #         indices += 2
    #         total = results.loc[indices]["total"]
    #         new_df["average"] = np.mean(total[-50:])
    #         new_df["deviation"] = np.std(total[-50:])
    #     new_df = pd.DataFrame(new_df).T
    #     new_df = new_df.set_index("file")
    #     df = pd.concat([df, new_df])
    # df.to_csv(os.path.join(parameters["file_path"], "evaluation.csv"))


def _parser():
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

    return parser


def parse_args():
    """
    Parse command line interface (CLI) arguments.

    :return: CLI arguments
    """
    args = _parser().parse_args()
    args.cpu_ids = [int(i) for i in args.cpu_ids]
    args.cpu_ids = [i for i in range(args.cpu_ids[0], args.cpu_ids[1] + 1)]
    return args


def main():
    args = parse_args()
    # single_run()
    parameter_sweep(args=args)


if __name__ == "__main__":
    main()
