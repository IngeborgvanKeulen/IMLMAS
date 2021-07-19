import json
import logging
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from q_agent.Agent import DQN, BootstrapDQN, SVM, RANDOM
from q_agent.Environment import BenchmarkEnvironment

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_action_space(params: Dict, div_amount: int, div_account: int, div_days: int, **kwargs) -> np.array:
    """
    Create the action space

    :param params: contains the minimums and stepsizes
    :param div_amount: the normalization value for amount and therefore as well the max value
    :param div_account: the normalization value for account and therefore as well the max value
    :param div_days: the normalization value for days and therefore as well the max value
    :return: the action space
    """
    # Set the defaults
    action_space = None

    # Create the action space
    for amount in range(params["min_amount"], div_amount + 1, params["step_amount"]):
        for account in range(params["min_account"], div_account + 1, params["step_account"]):
            for day in range(params["min_days"], div_days + 1, params["step_days"]):
                t = np.array([amount / div_amount, account / div_account, day / div_days])
                try:
                    action_space = np.vstack((action_space, np.hstack(t)))
                except:
                    action_space = t

    return action_space


def plot_and_save_results(rewards: List, days_uncaught: List, title: str, goal: int,
                          max_days: int, file_path: Optional[str] = None):
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


def store_results(params: Dict, final: List, days: List, df: pd.DataFrame, model):
    """
    Store the results from the complete simulation.

    :param params: the params used in the simulation
    :param final: the final rewards obtained in each episode
    :param days: the number of days of being uncaught in each episode
    :param df: a dataframe containing all actions taken with extra information like randomness
    :param model: if bootstrap, store each head
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

    # Save models
    if params["agent_model"] == "BDQN":
        model.store_heads(path=dir_path)


def run_games(params: Dict[str, Union[str, Dict]]):
    """
    This function runs the games for the given parameters and stores the results at the end.

    :param params: a dictionary containing all different parameters for all different modules.
    """
    # Create the action space; initialize the environment; initialize the DQN model
    action_space = create_action_space(params=params["action_space"], **params["environment"])
    env = BenchmarkEnvironment(action_space=action_space, **params["environment"],
                               model=params["agent_model"], max_days=params["simulation"]["max_days"])

    if params["agent_model"] == "DQN":
        model = DQN(action_dim=env.action_space.shape[0], state_dim=2, **params["model"], verbose=0)
    elif params["agent_model"] == "BDQN":
        model = BootstrapDQN(action_dim=env.action_space.shape[0], state_dim=2, **params["model"], verbose=0)
    elif params["agent_model"] == "SVM":
        model = SVM(**params["model"])
    elif params["agent_model"] == "RANDOM":
        model = RANDOM(**params["model"])
    else:
        raise ValueError(f"agent model is unknown {params['agent_model']}: options are DQN, BDQN, SVM")

    # Run the simulation
    final, days, df = model.run_learning(env=env, params=params["simulation"])

    # Store the results
    store_results(params, final, days, df, model)


def parameter_sweep(args: Namespace):
    """
    This function runs the simulation for all different parameter settings in todo_params.
    At the end all simulation are evaluated by looking at the last 50 games.

    :param args: the jobs and cpu ids arguments
    """
    todo_params = []
    for n_days in [3]:
        for decay_type in ["linear"]:
            for model, e_epi, max_epi in [("BDQN", 1750, 2500), ("DQN", 1000, 2000), ("DQN", 1000, 2000), ("DQN", 1000, 2000)]:
                for sampling_type, prio_buff in [("random", False)]:
                    for _ in ["BDQN"]:
                        todo_params.append([{
                            "agent_model": model,
                            "model": {
                                # SVM
                                "kernel": "rbf",
                                "regularization": 1000,
                                "tol": 1e-3,
                                "gamma": "scale",
                                # DQN / BootstrapDQN
                                "loss": "mse",
                                "learning_rate": 0.05,
                                "hidden_dim": 128,
                                "double": True,
                                "n_heads": 10,

                                # Business rules
                                "brs": [3],
                            },
                            "environment": {
                                "div_amount": 7000,
                                "div_account": 15,
                                "div_days": n_days,
                                "div_reward": 1,
                            },
                            "action_space": {
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

                                # Plots
                                "goal": 80000,
                                "title": "Launder as much as possible",
                            },
                            "file_path": f"../../FINAL_RESULTS_UPDATED/Experiments/Experiment_final/{model}/business_rule_4/{n_days}_days/{decay_type}_{sampling_type}_{e_epi}_{max_epi}/",
                        }])

    conf_file = "./engine_conf.json"
    for i in tqdm(todo_params):
        with open(conf_file, "r") as rf:
            conf = json.load(rf)
        conf["process_alerts"]["result_path"] = i[0]["file_path"]
        json.dump(conf, open(conf_file, "w"), indent=2)

        run_games(i[0])

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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    parameter_sweep(args=args)


if __name__ == "__main__":
    main()
