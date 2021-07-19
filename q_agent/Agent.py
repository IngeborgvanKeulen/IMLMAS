import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import svm
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tqdm import tqdm
from sklearn.exceptions import NotFittedError

from q_agent.Environment import BenchmarkEnvironment

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DQN:
    """
    Deep Q Neural Network
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, loss: str,
                 learning_rate: float, double: bool, brs: List = None, verbose=0, **kwargs):
        """
        Initialize the Deep Q Neural Network

        :param state_dim: the number of features in a state, defines the shape of the input
        :param action_dim: the number of of possible actions, defines the shape of the output
        :param hidden_dim: the number of nodes in a hidden layer
        :param learning_rate: the learning rate
        :param double: whether two models, one for training and one for predicting, should be used
        :param brs: the business rules to use
        :param verbose: whether training output should be shown in the console
        """
        self.brs = brs
        # Build and compile the model
        self.model = keras.Sequential([
            Dense(hidden_dim, activation=tf.nn.leaky_relu, input_shape=[state_dim],
                  kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
            Dense(hidden_dim * 2, activation=tf.nn.leaky_relu,
                  kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
            Dense(action_dim, activation=keras.activations.linear,
                  kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
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
                      kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
                Dense(hidden_dim * 2, activation=tf.nn.leaky_relu,
                      kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
                Dense(action_dim, activation=keras.activations.linear,
                      kernel_initializer=tf.initializers.random_uniform(minval=1, maxval=10)),
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
        :return: the memory buffer that might have been changed
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
            states = np.array([x[0] for x in batch])
            actions = [x[1] for x in batch]
            next_states = np.array([x[2] for x in batch])
            rewards = [x[3] for x in batch]
            not_dones = [float(not x[4]) for x in batch]
            days = [x[5] for x in batch]
            next_days = [x[6] for x in batch]

            # Predict the q-values for the next states; calculate the rewards
            # todo: norm possible to that day or in total
            next_states[:, 0] = next_states[:, 0] / (np.array(next_days) * 7000 * 13)
            states[:, 0] = states[:, 0] / (7000 * 13 * np.array(days))
            q_values_next = self.predict(next_states, use_target=True)
            rewards = rewards + gamma * np.max(q_values_next, axis=1) * not_dones

            # Predict the q-values for the current states; set the actual targets
            q_values = self.predict(state=states, use_target=False)
            q_values[np.arange(0, len(q_values)), actions] = rewards

            self.update(states=states, targets=q_values, epochs=epochs)
        return memory

    def run_learning(self, env: BenchmarkEnvironment, params: Dict)\
            -> Tuple[List, List, pd.DataFrame]:
        """
        Use q-learning in the given environment.

        :param env: the simulation environment
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
        for episode in tqdm(range(params["max_episodes"])):
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
                    eps_episode = max((0.01 - 1) * ep / params["e_episodes"] + 1, 0.01)
                elif params["eps_decay"] == "annealing":
                    eps_episode = max(pow(params["epsilon_factor"], ep), 0.01)
                else:
                    eps_episode = params["epsilon"]

            else:
                # Obtain epsilon for this episode
                if params["eps_decay"] == "linear":
                    eps_episode = max((0.01 - 1) * episode / params["e_episodes"] + 1, 0.01)
                elif params["eps_decay"] == "annealing":
                    eps_episode = max(pow(params["epsilon_factor"], episode), 0.01)
                else:
                    eps_episode = params["epsilon"]

            # Update the target model when using a separate one and n_update steps have passed
            if self.double and episode % params["n_update"] == 0:
                self.target_update()

            day = 0
            # Run for the maximum number of days
            while day < max_days:

                # Either explore or exploit
                if random.random() < eps_episode or len(memory) < params["replay_size"]:
                    # The taken action is random
                    is_random = True

                    # Select completely random an valid action
                    valid = round(state[-1] * env.max_days) + env.action_space[:, -1] * env.div_days
                    options = np.arange(len(env.action_space))[valid <= max_days]
                    if len(options) == 0:
                        raise ValueError("No options available (should be impossible)")
                    action = random.choice(options)

                else:
                    # The taken action is not random
                    is_random = False

                    # Find the best action
                    norm_state = state.copy()
                    if day != 0:
                        # todo: how to do norm
                        norm_state = norm_state / [env.div_amount * 13 * day, 1]
                    q_values = self.predict(state=norm_state, use_target=False)
                    action = np.argmax(q_values)
                    # In case this action is invalid, find the next-best action that is valid
                    while round(state[-1] * env.max_days) + round(env.action_space[action][-1] * env.div_days) > max_days:
                        q_values[0][action] = -1
                        action = np.argmax(q_values)
                        if max(q_values[0]) == -1:
                            break

                # Perform the chosen action
                next_state, reward, done = env.step(brs=self.brs, state=state, action=action)

                if round(next_state[-1] * env.max_days) == day:
                    raise ValueError("Day did not change")
                # When the current day is the max number of days (or greater than), then the game is over
                day = round(next_state[-1] * env.max_days)
                if day >= max_days:
                    done = True
                    if day > max_days:
                        reward = 0
                        raise ValueError("Days surpass max")

                # Update the total rewards; add experience to the memory; obtain q_values for the current state
                total += reward if reward > 0 else 0
                next_state[0] = total
                cur_day = round(state[-1] * env.max_days)
                memory.append((state, action, next_state, reward, done, cur_day if cur_day != 0 else 1, day))
                if not params["replay"]:
                    q_values = self.predict(
                        state=state / [env.div_amount * env.div_account * round(state[-1] * env.max_days), 1],
                        use_target=False)

                # Add the current action and context information to the dataframe
                df = df.append({"episode": episode,
                                "eps_episode": eps_episode,
                                "day": day,
                                "state": state *
                                         [env.div_amount * env.div_account * round(state[-1] * env.max_days),
                                          env.max_days],
                                "action": env.action_space[action] * [env.div_amount, env.div_account, env.div_days],
                                "random": is_random,
                                "reward": reward,
                                "next_state": next_state * [env.div_amount * env.div_account * day, env.max_days],
                                "total": total}, ignore_index=True)
                if len(df) > 1:
                    if reward < 0:
                        if df["action"].iloc[-1][0] == 0:
                            print(df)
                            raise ValueError("AMOUNT IS ZERO BUT REWARD NEGATIVE")
                        elif self.brs == [4]:
                            if df["episode"].iloc[-2] == df["episode"].iloc[-1]:
                                if df["action"].iloc[-1][0] * (df["action"].iloc[-1][1] - 2) < 40000:
                                    if df["action"].iloc[-2][0] == 0:
                                        print(df)
                                        raise ValueError("ACCORDING TO BR 4 SHOULD BE FINE")

                if done:
                    # The run is done, so optionally train on the last sample and then break
                    if not params["replay"]:
                        q_values[0][action] = reward
                        self.update(states=state, targets=q_values, epochs=epochs)
                    else:
                        memory = self.replay(memory=memory, size=params["replay_size"], gamma=params["gamma"],
                                             epochs=epochs, finite=params["finite"], sample_method=params["sampling"],
                                             prio_buffer=params["prio_buffering"])
                    break

                if params["replay"]:
                    # Train the model by replaying a batch of the memory
                    memory = self.replay(memory=memory, size=params["replay_size"], gamma=params["gamma"],
                                         epochs=epochs, finite=params["finite"], sample_method=params["sampling"],
                                         prio_buffer=params["prio_buffering"])
                else:
                    # Train the model by using the last sample only
                    q_values_next = self.predict(state=next_state, use_target=True)
                    q_values[0][action] = reward + params["gamma"] * np.max(q_values_next)
                    self.update(states=state, targets=q_values, epochs=epochs)

                # Go to the next state
                state = next_state

            # Add the results and optionally plot them
            days_uncaught.append(day)
            final.append(total)

        return final, days_uncaught, df


class BootstrapDQN:
    """
    Bootstrapped Deep Q Neural Network
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, loss: str,
                 learning_rate: float, double: bool, n_heads: int, brs: List = None, verbose=0, **kwargs):
        """
        Initialize the Deep Q Neural Network

        :param state_dim: the number of features in a state, defines the shape of the input
        :param action_dim: the number of of possible actions, defines the shape of the output
        :param hidden_dim: the number of nodes in a hidden layer
        :param learning_rate: the learning rate
        :param double: whether two models, one for training and one for predicting, should be used
        :param brs: the business rules to use
        :param verbose: whether training output should be shown in the console
        """
        self.brs = brs
        reload_path = None
        # reload_path = "/home/ingeborg.local/Thesis/FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_4/3_days/setting_1000_2/head_0.h5"

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
                      kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=1)),
                Dense(hidden_dim * 2, activation=tf.nn.leaky_relu,
                      kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=1)),
            ])
            self.target_heads = []
            for i in range(n_heads):
                self.target_heads.append(tf.keras.Sequential(
                    [target, Dense(action_dim, activation=keras.activations.linear)]))
                self.target_heads[i].compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                             loss=loss,
                                             metrics=['mae', 'mse'])
                self.target_update(idx=i)

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

            # todo: how to do norm
            next_states[:, 0] = next_states[:, 0] / (np.array(next_days) * 7000 * 13)
            states[:, 0] = states[:, 0] / (7000 * 13 * np.array(days))

            def train_each_head(head_idx):
                head_idx = head_idx[0]
                mask = masks[:, head_idx]
                next_states_c = tf.convert_to_tensor(next_states[mask == 1])
                states_c = tf.convert_to_tensor(states[mask == 1])

                # Predict the q-values for the next states; calculate the rewards
                q_values_next = self.predict(next_states_c, idx=head_idx, use_target=True)
                rewards_t = rewards[mask == 1] + gamma * np.max(q_values_next, axis=1).T * not_dones[mask == 1]

                # Predict the q-values for the current states; set the actual targets
                q_values = self.predict(state=states_c, use_target=False, idx=head_idx).numpy()
                q_values[np.arange(0, len(q_values)), actions[mask == 1]] = rewards_t

                self.update(states=states_c, targets=q_values, epochs=epochs, idx=head_idx)

            head_list = list(np.arange(len(self.heads)))
            random.shuffle(head_list)
            for i in head_list:
                train_each_head([i])

        return memory

    def run_learning(self, env: BenchmarkEnvironment, params: Dict):
        """
        Use an q-learning in the given environment

        :param env: the simulation environment
        :param params: the parameters for the simulation
        :return: the final rewards and days without being caught per episode
        """
        df = pd.DataFrame([], columns=["episode", "head", "eps_episode", "day", "state", "action",
                                       "random", "reward", "next_state", "total"])
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
            head_idx = random.randint(0, len(self.heads) - 1)

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
            if self.double and episode % params["n_update"] == 0:
                for i in range(len(self.heads)):
                    self.target_update(idx=i)

            day = 0
            # Run for the maximum number of days
            while day < params["max_days"]:
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
                        norm_state = norm_state / [env.div_amount * 13 * day, 1]
                    norm_state = tf.convert_to_tensor(norm_state.reshape(1, -1))
                    q_values = self.predict(state=norm_state, idx=head_idx, use_target=False).numpy()

                    # First find the best action
                    action = np.argmax(q_values)
                    while round(state[-1] * env.max_days) + \
                            round(env.action_space[action][-1] * env.div_days) > params["max_days"]:
                        q_values[0][action] = -1
                        action = np.argmax(q_values)
                        if max(q_values[0]) == -1:
                            logger.info("All options are negative")
                            is_random = "all options are negative"
                            break

                # Perform the chosen action; if this is the last day then done should be set to True
                next_state, reward, done = env.step(brs=self.brs, state=state, action=action)

                if round(next_state[-1] * env.max_days) == day:
                    raise ValueError("Day did not change")
                day = round(next_state[-1] * env.max_days)
                if day == params["max_days"]:
                    done = True
                elif day > params["max_days"]:
                    raise ValueError("Surpass max days")
                    done = True
                    reward = 0

                # Update the total rewards; add experience to the memory; obtain q_values for the current state
                total += reward if reward > 0 else 0

                next_state[0] = total
                cur_day = round(state[-1] * env.max_days)

                memory.append((state, action, next_state, reward, done,
                               self.random_state.binomial(1, params["probability"], len(self.heads)),
                               cur_day if cur_day != 0 else 1, day))
                if not params["replay"]:
                    q_values = self.predict(
                        state=state / [env.div_amount * env.div_account * round(state[-1] * env.max_days), 1],
                        idx=head_idx,
                        use_target=False)

                # Add the current action and context information to the dataframe
                df = df.append({"episode": episode,
                                "head": head_idx,
                                "eps_episode": eps_episode,
                                "day": day,
                                "state": state * [1, env.max_days],
                                "action": env.action_space[action] * [env.div_amount, env.div_account, env.div_days],
                                "random": is_random,
                                "reward": reward,
                                "next_state": next_state * [1, env.max_days],
                                "total": total}, ignore_index=True)

                if len(df) > 1:
                    if reward < 0:
                        if df["action"].iloc[-1][0] == 0:
                            print(df)
                            raise ValueError("AMOUNT IS ZERO BUT REWARD NEGATIVE")
                        elif self.brs == [4]:
                            if df["episode"].iloc[-2] == df["episode"].iloc[-1]:
                                if df["action"].iloc[-1][0] * (df["action"].iloc[-1][1] - 2) < 40000:
                                    if df["action"].iloc[-2][0] == 0:
                                        print(df)
                                        raise ValueError("ACCORDING TO BR 4 SHOULD BE FINE")

                if done:
                    # The run is done, so optionally train on the last sample and then break
                    if not params["replay"]:
                        q_values[0][action] = reward
                        self.update(states=state, targets=q_values, epochs=epochs, idx=head_idx)
                    else:
                        memory = self.replay(memory=memory, size=params["replay_size"], gamma=params["gamma"],
                                             epochs=epochs, finite=params["finite"], sample_method=params["sampling"],
                                             prio_buffer=params["prio_buffering"])
                    break

                if params["replay"]:
                    # Train the model by replaying a batch of the memory
                    # todo: train every model or only current one?
                    memory = self.replay(memory=memory, size=params["replay_size"], gamma=params["gamma"],
                                         epochs=epochs, finite=params["finite"], sample_method=params["sampling"],
                                         prio_buffer=params["prio_buffering"])
                else:
                    # Train the model by using the last sample only
                    q_values_next = self.predict(state=next_state, idx=head_idx, use_target=True)
                    q_values[0][action] = reward + params["gamma"] * np.max(q_values_next)
                    self.update(states=state, targets=q_values, epochs=epochs, idx=head_idx)

                # Go to the next state
                state = np.copy(next_state)

            # Add the results and plot them
            days_uncaught.append(day)
            final.append(total)

        return final, days_uncaught, df


class SVM:
    """
    Support Vector Machine
    """

    def __init__(self, kernel: str = "rbf", regularization: int = 1000,
                 tol: float = 1e-3, gamma: str = "scale", brs: List = None, **kwargs):
        """
        Initialize the SVM

        :param kernel: the kernel the SVM should use
        :param regularization: the regularization parameter defining the importance of preventing misclassifications
        :param tol: the error tolerance before stopping with training
        :param gamma: the gamma value
        :param brs: the business rules to use
        """
        # Set the model
        self.model = svm.SVC(kernel=kernel, C=regularization, tol=tol, gamma=gamma)
        self.brs = brs

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

    def run_learning(self, env: BenchmarkEnvironment, params: Dict) \
            -> Tuple[List, List, pd.DataFrame]:
        """
        Use svm learning in the given environment.

        :param env: the simulation environment
        :param params: the parameters for the simulation
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
                        raise ValueError("No options available (should be impossible)")
                    action = random.choice(options)

                else:
                    # The taken action is not random
                    is_random = False

                    # Find the best action
                    try:
                        q_values = self.predict(action=env.action_space)
                        exp_rew = env.action_space[:, 0] * env.div_amount * env.action_space[:, 1] * env.div_account \
                                  / (env.action_space[:, 2] * env.div_days)
                        action = np.argmax(q_values * exp_rew)

                        # In case this action is invalid, find the next-best action that is valid
                        while day + round(env.action_space[action][-1] * env.div_days) > max_days:
                            q_values[action] = -1
                            action = np.argmax(q_values * exp_rew)
                            if max(q_values) == -1:
                                break

                    except NotFittedError:
                        # The taken action is random
                        is_random = True

                        # Select completely random an valid action
                        valid = day + env.action_space[:, -1] * env.div_days
                        options = np.arange(len(env.action_space))[valid <= max_days]
                        if len(options) == 0:
                            raise ValueError("No options available (should be impossible)")
                        action = random.choice(options)

                # Perform the chosen action
                next_state, reward, done = env.step(brs=self.brs, state=state, action=action)

                # When the current day is the max number of days (or greater than), then the game is over
                day = round(next_state[-1] * env.max_days)
                if day >= max_days:
                    if day > max_days and done is False:
                        reward = 0
                    done = True

                # Update the total rewards; add experience to the memory; obtain q_values for the current state
                total += 0 if reward == -1 else reward * env.div_reward

                # Add the current action and context information to the dataframe
                temp = round(state[-1] * env.max_days)
                df = df.append({"episode": episode,
                                "eps_episode": eps_episode,
                                "day": day,
                                "state": state * [env.div_amount * env.div_account * temp, env.max_days],
                                "action": env.action_space[action] * [env.div_amount, env.div_account, env.div_days],
                                "random": is_random,
                                "reward": reward,
                                "next_state": next_state * [env.div_amount * env.div_account * temp, env.max_days],
                                "total": total}, ignore_index=True)

                if reward != -1:
                    reward = 1
                memory.append((env.action_space[action], reward))

                if params["replay"]:
                    # Train the model by replaying a batch of the memory
                    memory = self.replay(memory=memory, finite=params["finite"], prio_buffer=params["prio_buffering"])
                else:
                    # Train the model by using the last sample only
                    self.update(states=state, targets=[reward])

                if done:
                    break

                # Go to the next state
                state = next_state

            # Add the results and optionally plot them
            days_uncaught.append(day)
            final.append(total)

        return final, days_uncaught, df


class RANDOM:
    """
    Random
    """

    def __init__(self, brs: List = None, **kwargs):
        """
        Initialize the random model

        :param brs: the business rules to use
        """
        # Set the brs
        self.brs = brs

    def run_learning(self, env: BenchmarkEnvironment, params: Dict) \
            -> Tuple[List, List, pd.DataFrame]:
        """
        Use svm learning in the given environment.

        :param env: the simulation environment
        :param params: the parameters for the simulation
        :return: the final rewards, days without being caught per episode and all taken actions
        """
        # Set the most used parameters
        max_days = params["max_days"]

        # Initialize the memory buffer; rewards; days per episode; dataframe for all actions
        final = []
        days_uncaught = []
        df = pd.DataFrame([], columns=["episode", "day", "state", "action",
                                       "random", "reward", "next_state", "total"])

        # Run for the given number of episodes
        for episode in tqdm(range(params["max_episodes"])):
            # Reset the environment
            state = env.reset()
            total = 0

            day = 0
            # Run for the maximum number of days
            while day < max_days:
                # The taken action is random
                is_random = True

                # Select completely random an valid action
                valid = day + env.action_space[:, -1] * env.div_days
                options = np.arange(len(env.action_space))[valid <= max_days]
                if len(options) == 0:
                    raise ValueError("No options available (should be impossible)")
                action = random.choice(options)

                # Perform the chosen action
                next_state, reward, done = env.step(brs=self.brs, state=state, action=action)

                # When the current day is the max number of days (or greater than), then the game is over
                day = round(next_state[-1] * env.max_days)
                if day >= max_days:
                    if day > max_days and done is False:
                        reward = 0
                    done = True

                # Update the total rewards; add experience to the memory; obtain q_values for the current state
                total += 0 if reward < 0 else reward * env.div_reward

                # Add the current action and context information to the dataframe
                temp = round(state[-1] * env.max_days)
                df = df.append({"episode": episode,
                                "day": day,
                                "state": state * [env.div_amount * env.div_account * temp, env.max_days],
                                "action": env.action_space[action] * [env.div_amount, env.div_account, env.div_days],
                                "random": is_random,
                                "reward": reward,
                                "next_state": next_state * [env.div_amount * env.div_account * temp, env.max_days],
                                "total": total}, ignore_index=True)

                if done:
                    break

                # Go to the next state
                state = next_state

            # Add the results and optionally plot them
            days_uncaught.append(day)
            final.append(total)

        return final, days_uncaught, df
