import os

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from visualizations.add_edge import add_edge


SHOW_ALL = False


def create_network(input_path="../AMLSim/outputs/ml_agent/transactions.csv", k=0.5, iterations=80):
    transactions_df = pd.read_csv(input_path, header=0)

    # Rename columns to the name the engine expects
    transactions_df = transactions_df.rename(columns={"tran_id": "transaction_id",
                                                      "tran_timestamp": "paid_at",
                                                      "base_amt": "amount",
                                                      "orig_acct": "sender",
                                                      "bene_acct": "receiver"
                                                      })

    bad, main = get_bad()
    # First create the layout with networkx
    G = nx.MultiDiGraph()
    for index, row in transactions_df.iterrows():
        if SHOW_ALL:
            G.add_edge(row["sender"], row["receiver"])
        else:
            if row["sender"] in bad or row["receiver"] in bad:
                G.add_edge(row["sender"], row["receiver"])

    pos = nx.spring_layout(G, k=k, iterations=iterations)  # positions for all nodes

    return G, pos


def get_bad(input_path="../AMLSim/outputs/ml_agent/"):
    bad_df = pd.read_csv(os.path.join(input_path, "alert_accounts.csv"), header=0)
    main_df = pd.read_csv(os.path.join(input_path, "main_accounts.csv"), header=0)

    return set(bad_df["acct_id"].values), set(main_df["MAIN_ACCOUNT_ID"].values)


def visualize_network(G, pos):
    node_size = 10
    line_width = 1
    bad, main = get_bad()

    # Now create the interactive graph
    fig = go.Figure()

    # Create the edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        start = pos[edge[0]]
        end = pos[edge[1]]
        edge_x, edge_y = add_edge(start, end, edge_x, edge_y, dot_size=node_size)
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=line_width, color='black'),
        hoverinfo='none',
        name="transaction",
        mode='lines',
        showlegend=True))

    # Create the nodes per entity; normal, bad, main
    node_x = [[], [], []]
    node_y = [[], [], []]
    node_text = [[], [], []]
    for idx, node in enumerate(G):
        x, y = pos[node]

        if node in main:
            node_x[2].append(x)
            node_y[2].append(y)
            node_text[2].append("main: " + str(node))
        elif node in bad:
            node_x[1].append(x)
            node_y[1].append(y)
            node_text[1].append("bad: " + str(node))
        else:
            node_x[0].append(x)
            node_y[0].append(y)
            node_text[0].append("normal: " + str(node))

    # Plot the nodes in the figure
    colours = ["lightblue", "pink", "crimson"]
    groups = ["group1", "group2", "group3"]
    names = ["normal", "sar account", "main account"]
    for idx in range(3):
        fig.add_trace(go.Scatter(
            x=node_x[idx], y=node_y[idx],
            legendgroup=groups[idx],
            name=names[idx],
            mode="markers",
            hoverinfo='text',
            text=node_text[idx],
            marker=dict(
                showscale=False,
                colorscale='Jet',
                reversescale=True,
                color=colours[idx],
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2)))

    fig.update_layout(title='<br>Demo',
                      titlefont_size=16,
                      showlegend=True,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
                      plot_bgcolor='rgba(0,0,0,0)',
                      width=1500,
                      height=1200)

    fig.write_html("output.html")


def main() -> None:
    G, pos = create_network()
    visualize_network(G, pos)


if __name__ == "__main__":
    main()
