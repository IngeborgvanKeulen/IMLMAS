import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check_validity(file_path, results):
    # Due to a mistake with int vs round, check if next_state day is never the same as day (otherwise it is invalid)
    temp = results[["state", "next_state"]]
    temp["state"] = [float(i.split(" ")[1][:-2]) for i in temp["state"].values]
    temp["next_state"] = [float(i.split(" ")[1][:-2]) for i in temp["next_state"].values]

    if sum(temp["state"] == temp["next_state"]) != 0:
        print("DAY STATE SAME AS NEXT STATE: INVALID!!! ", file_path)
        raise ValueError("INVALIDDDD")


def plot_normal(dir_path, days=3, br=None):
    file_paths = os.listdir(dir_path)

    df = pd.DataFrame([])
    for file in file_paths:
        # Skip evaluation files
        if file == "evaluation.csv":
            continue

        # Get the file path
        file_path = os.path.join(dir_path, file)

        # Get the action-states file; only keep the last reward of each game; get the number of heads
        results = pd.read_csv(os.path.join(file_path, "action_states.csv"))
        check_validity(file_path, results)
        results = results.drop_duplicates(subset=["episode"], keep="last")
        results.reset_index(inplace=True)
        total = results["total"]
        df = pd.concat([df, total], axis=1, ignore_index=True)

    df["min"] = df.min(axis=1)
    df["max"] = df.max(axis=1)
    df["mean"] = df.mean(axis=1)

    plt.figure()
    plt.plot(np.arange(len(df)), df["mean"].rolling(window=25).mean())
    dev = df["mean"].rolling(window=25).std()
    plt.fill_between(np.arange(len(df)), df["mean"].rolling(window=25).mean() - dev,
                     df["mean"].rolling(window=25).mean() + dev, facecolor='lightblue',
                     interpolate=True)
    # plt.fill_between(np.arange(len(df)), df["min"].rolling(window=25).min(), df["max"].rolling(window=25).max(), facecolor='lightblue',
    #                  interpolate=True)
    if br is None:
        br = int(dir_path[-1])
    if br == 1:
        # plt.ylim(-1000, round((4000*13) * days, -4))
        plt.ylim(-1000, (4000 * 13) * days + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"First business rule, {days} days")
        plt.axhline(y=(4000 * 13) * days, c="r")
    elif br == 2:
        plt.ylim(-1000, (7000 * 6) * days + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"Second business rule, {days} days")
        plt.axhline(y=(7000 * 6) * days, c="r")
    elif br == 3:
        plt.ylim(-1000, 22000 * days + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"Third business rule, {days} days")
        plt.axhline(y=22000 * days, c="r")
    elif br == 4:
        plt.ylim(-1000, 39000 * (days - 1) + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"Fourth business rule, {days} days")
        plt.axhline(y=39000 * (days - 1), c="r")
    elif br == 12:
        plt.ylim(-1000, (4000 * 6) * days + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"First and second business rule, {days} days")
        plt.axhline(y=(4000 * 6) * days, c="r")

    plt.savefig(os.path.join(file_path, "reward_over_episodes_averaged.png"))
    plt.show()

    plt.figure()
    for i in range(len(file_paths)):
        plt.plot(np.arange(len(df)), df.iloc[:, i].rolling(window=25).mean())

    if br is None:
        br = int(dir_path[-1])
    if br == 1:
        # plt.ylim(-1000, round((4000*13) * days, -4))
        plt.ylim(-1000, (4000 * 13) * days + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"First business rule, {days} days")
        plt.axhline(y=(4000 * 13) * days, c="r")
    elif br == 2:
        plt.ylim(-1000, (7000 * 6) * days + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"Second business rule, {days} days")
        plt.axhline(y=(7000 * 6) * days, c="r")
    elif br == 3:
        plt.ylim(-1000, 22000 * days + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"Third business rule, {days} days")
        plt.axhline(y=22000 * days, c="r")
    elif br == 4:
        plt.ylim(-1000, 39000 * (days - 1) + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"Fourth business rule, {days} days")
        plt.axhline(y=39000 * (days - 1), c="r")
    elif br == 12:
        plt.ylim(-1000, (4000 * 6) * days + 5000)
        plt.xlim(-50, last_ep + 50)
        plt.title(f"First and second business rule, {days} days")
        plt.axhline(y=(4000 * 6) * days, c="r")

    plt.savefig(os.path.join(file_path, "reward_over_episodes.png"))
    plt.show()


def evaluate_normal(dir_path):
    # Get all the directories in the given path
    dirs = os.listdir(dir_path)

    # Initialize empty dataframe and fill it
    df = pd.DataFrame([])
    for file_dir in dirs:
        if not file_dir.startswith("setting"):
            continue
        # Add information about the experiment settings
        params = pd.read_json(os.path.join(dir_path, file_dir, "params.json"), orient="index")
        new_df = params.loc["simulation"]
        new_df = new_df.append(pd.Series([file_dir], index=["file"]))
        new_df = new_df.dropna()

        # Obtain the rewards; drop random games; only keep the last round of each game
        results = pd.read_csv(os.path.join(dir_path, file_dir, "action_states.csv"))
        random_games = np.unique(results.loc[results["random"] == True]["episode"])
        results = results.loc[~ results["episode"].isin(random_games)]
        results = results.drop_duplicates(subset=["episode"], keep="last")

        # Calculate the average and deviation from the last 50 games
        new_df["average"] = np.mean(results["total"][-50:])
        new_df["deviation"] = np.std(results["total"][-50:])

        # Combine everything in the dataframe
        new_df = pd.DataFrame(new_df).T
        new_df = new_df.set_index("file")
        df = pd.concat([df, new_df])

    df = df.sort_values(["episodes", "replay_size", "n_update", "eps_decay", "stretched"])
    df.to_csv(os.path.join(dir_path, "evaluation.csv"))


def plot_bootstrap(dir_path, days=3, br=None):
    file_paths = os.listdir(dir_path)

    for file in file_paths:
        # Skip evaluation files
        if not file.startswith("setting"):
            continue

        # Get the file path
        file_path = os.path.join(dir_path, file)

        # Get the action-states file; only keep the last reward of each game; get the number of heads
        results = pd.read_csv(os.path.join(file_path, "action_states.csv"))
        check_validity(file_path, results)
        last_ep = max(results["episode"])
        results = results.drop_duplicates(subset=["episode"], keep="last")
        results.reset_index(inplace=True)
        heads = np.unique(results["head"])

        # Create the figure
        plt.figure(figsize=(12, 8), dpi=100)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=25)
        for i in heads:
            df = results.loc[results["head"] == i]
            # todo: which way of averaging is best?
            # plt.plot(df["episode"], df["total"], label="normal")
            # plt.plot(df["episode"], df["total"].rolling(window=25, center=False).mean(), label="rolling_mean")
            # plt.plot(df["episode"], df["total"].rolling(window=25, center=True).mean(), label="centered rolling_mean")
            # plt.plot(df["episode"], df["total"].rolling(window=25, min_periods=1, center=False).mean(),
            #          label="min_periods")
            plt.plot(df["episode"], df["total"].rolling(window=25, min_periods=1, center=True).mean(),
                     label=f"head {i+1}")
            # plt.plot(df["episode"], df["total"].rolling(window=indexer, min_periods=1, center=False).mean(),
            #          label="forward ")
        plt.legend(framealpha=1, loc=2)
        plt.xlabel("Episodes")
        plt.ylabel("Averaged total reward")

        if br is None:
            br = int(dir_path[-1])
        if br == 1:
            # plt.ylim(-1000, round((4000*13) * days, -4))
            plt.ylim(-1000, (4000*13) * days + 5000)
            plt.xlim(-50, last_ep + 50)
            plt.title(f"First business rule, {days} days")
            plt.axhline(y=(4000*13) * days, c="r")
        elif br == 2:
            plt.ylim(-1000, (7000 * 6) * days + 5000)
            plt.xlim(-50, last_ep + 50)
            plt.title(f"Second business rule, {days} days")
            plt.axhline(y=(7000 * 6) * days, c="r")
        elif br == 3:
            plt.ylim(-1000, 22000 * days + 5000)
            plt.xlim(-50, last_ep + 50)
            plt.title(f"Third business rule, {days} days")
            plt.axhline(y=22000 * days, c="r")
        elif br == 4:
            plt.ylim(-1000, 39000 * (days-1) + 5000)
            plt.xlim(-50, last_ep + 50)
            plt.title(f"Fourth business rule, {days} days")
            plt.axhline(y=39000 * (days-1), c="r")
        elif br == 12:
            plt.ylim(-1000, (4000 * 6) * days + 5000)
            plt.xlim(-50, last_ep + 50)
            plt.title(f"First and second business rule, {days} days")
            plt.axhline(y=(4000 * 6) * days, c="r")

        plt.savefig(os.path.join(file_path, "reward_over_episodes.png"))
        plt.show()


def evaluate_bootstrap(dir_path):
    # Get all the directories in the given path
    dirs = os.listdir(dir_path)

    # Initialize empty dataframe and fill it
    df = pd.DataFrame([])
    for file_dir in dirs:
        if not file_dir.startswith("setting"):
            continue

        # Add information about the experiment settings
        params = pd.read_json(os.path.join(dir_path, file_dir, "params.json"), orient="index")
        # new_df = params.loc["simulation"]
        new_df = params.loc["simulation"]
        try:
            new_df["number of heads"] = params.loc["model"]["n_heads"]
        except:
            new_df["number of heads"] = params.loc["model"].values[0]["n_heads"]
        new_df = new_df.append(pd.Series([file_dir], index=["file"]))
        new_df = new_df.dropna()

        # Obtain the rewards; drop random games; only keep the last round of each game
        results = pd.read_csv(os.path.join(dir_path, file_dir, "action_states.csv"))
        random_games = np.unique(results.loc[results["random"] == True]["episode"])
        results = results.loc[~ results["episode"].isin(random_games)]
        results = results.drop_duplicates(subset=["episode"], keep="last")

        # Calculate the average and deviation from the last 50 games
        new_df["average"] = np.mean(results["total"][-50:])
        new_df["deviation"] = np.std(results["total"][-50:])
        new_df["best_average"] = new_df["average"]
        new_df["best_deviation"] = new_df["deviation"]

        for i in range(int(new_df["number of heads"])):
            av = np.mean(results.loc[results["head"] == i]["total"][-50:])
            std = np.std(results.loc[results["head"] == i]["total"][-50:])
            if av > new_df["best_average"]:
                new_df["best_average"] = av
                new_df["best_deviation"] = std

        # Combine everything in the dataframe
        new_df = pd.DataFrame(new_df).T
        new_df = new_df.set_index("file")
        df = pd.concat([df, new_df])

    # df = df.sort_values(["episodes", "number of heads", "probability"])
    df.to_csv(os.path.join(dir_path, "evaluation.csv"))


# bootstrap_path = "../../FINAL_RESULTS/BOOTSTRAP/"
# bootstrap_path = "../../FINAL_RESULTS/BOOTSTRAP/03_05/"
# bootstrap_br_paths = os.listdir(bootstrap_path)
# for p in bootstrap_br_paths:
#     plot_bootstrap(dir_path=os.path.join(bootstrap_path, p))
#     evaluate_bootstrap(dir_path=os.path.join(bootstrap_path, p))


###################### experiment 1 ######################
# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_3/BOOT/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_3/BOOT/", br=3)

# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_4/BOOT/heads_5/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_4/BOOT/heads_5/", br=4)

# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_4/BOOT/heads_10/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_4/BOOT/heads_10/", br=4)

# plot_normal(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_1/DQN/")
# evaluate_normal(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_1/DQN/")

# plot_normal(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_3/DQN/")
# evaluate_normal(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_3/DQN/")

# plot_normal(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_4/DQN/episodes_2000/")
# evaluate_normal(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_4/DQN/episodes_2000/")

# plot_normal(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_4/DQN/episodes_3000/")
# evaluate_normal(dir_path="../../FINAL_RESULTS/Experiments/Experiment_1/business_rule_4/DQN/episodes_3000/")


###################### experiment 2 ######################
# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_2/business_rule_2/BOOT/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_2/business_rule_2/BOOT/", br=2)

# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_2/business_rule_1/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_2/business_rule_1/", br=1)

# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_2/business_rule_1_2/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_2/business_rule_1_2/", br=12)

###################### experiment 4 ######################
# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_4/3_days/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_4/3_days/", days=3, br=4)

evaluate_bootstrap(dir_path="../../FINAL_RESULTS/tmp/15_05_reload_br_2_7_days/")
plot_bootstrap(dir_path="../../FINAL_RESULTS/tmp/15_05_reload_br_2_7_days/", days=7, br=2)

# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_2/r_days/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_2/3_days/", br=2)
# bootstrap_path = "../../FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_2/3_days/"
# bootstrap_br_paths = os.listdir(bootstrap_path)
# for p in bootstrap_br_paths:
#     if p.startswith("episodes"):
#         plot_bootstrap(dir_path=os.path.join(bootstrap_path, p), br=2)

# evaluate_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_2/7_days/episode_sweep/")
# plot_bootstrap(dir_path="../../FINAL_RESULTS/Experiments/Experiment_4/BOOT/business_rule_2/7_days/episode_sweep/", days=7, br=2)

# normal_path = "../../FINAL_RESULTS/dqn_sweep_1_br_26_04_2021/"
# plot_normal(dir_path=normal_path)
# evaluate_normal(dir_path=normal_path)

# normal_br_paths = os.listdir(normal_path)
# for p in normal_br_paths:
#     plot_normal(dir_path=os.path.join(normal_path, p))
#     evaluate_normal(dir_path=os.path.join(normal_path, p))
