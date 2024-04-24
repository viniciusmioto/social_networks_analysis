#!./venv/bin/python

# -------------------- ABOUT --------------------#
#
# Author: Vinícius Mioto
# Professor: André Luís Vignatti
# BSc. Computer Science
# Universidade Federal do Paraná
#
# -----------------------------------------------#


# -------------------- LIBRARIES --------------------#

import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statistics as sts
from networkx.algorithms import isomorphism
import itertools
import random


# -------------------- FUNCTIONS --------------------#


def read_graph_from_edge_list(file_path):
    """Reads a directed graph from an edge list file."""
    G = nx.DiGraph()
    with open(file_path, "r") as file:
        for line in file:
            edge = line.strip().split()
            if len(edge) == 2:
                source, target = edge
                G.add_edge(source, target)
    return G


def generate_subgraphs(graph, size):
    """
    Generate all subgraphs of a given size from a graph.

    Parameters:
        graph (NetworkX graph): The input graph from which subgraphs are generated.
        size (int): The size of subgraphs to generate (number of nodes).

    Returns:
        list: List of subgraphs of the specified size.
    """
    print("Generating subgraphs of size", size)
    subgraphs = []
    for nodes in itertools.combinations(graph.nodes(), size):
        # guarantee that there is no isolated node
        subgraph = graph.subgraph(nodes)
        if nx.is_weakly_connected(subgraph):
            subgraphs.append(subgraph)

    print("Generated", len(subgraphs), "subgraphs")
    return subgraphs


def subgraph_count(graph, motifs):
    """
    Count the occurrences of a list of subgraphs within a given graph.

    Parameters:
        graph (NetworkX graph): The input graph in which occurrences are counted.
        motifs (list): List of subgraphs whose occurrences are being counted.

    Returns:
        dict: A dictionary containing the counts of occurrences for each motif.
    """
    # Generate all subgraphs of the largest size in the list of motifs
    max_size = max([subgraph.number_of_nodes() for subgraph in motifs])
    all_subgraphs = generate_subgraphs(graph, max_size)

    # Initialize a dictionary to store counts for each motif
    motif_counts = {i: 0 for i, motif in enumerate(motifs)}

    # Iterate through all subgraphs and motifs to count occurrences
    for subgraph in all_subgraphs:
        for i, motif in enumerate(motifs):
            if len(subgraph.edges()) == len(motif.edges()):
                if nx.is_isomorphic(subgraph, motif):
                    motif_counts[i] += 1

    return motif_counts


def generate_sample_graph(original_graph, sample_size):
    """
    Generates a random sample of a graph.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_size (int): The number of nodes to sample.

    Returns:
        NetworkX graph: A random sample of the input graph.
    """

    sample_graph = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    # Convert nodes to list for compatibility with random.sample
    nodes_list = list(original_graph.nodes())

    # Sample nodes
    sample_nodes = random.sample(nodes_list, min(sample_size, len(original_graph)))
    sample_graph.add_nodes_from(sample_nodes)

    # Sample edges
    for u, v in original_graph.edges():
        if u in sample_nodes and v in sample_nodes:
            sample_graph.add_edge(u, v)

    return sample_graph


def generate_configuration_model_graph(original_graph, seed):
    """
    Generate a random graph using the configuration model while preserving the properties of an existing graph.

    Ignore any self-loops or parallel edges in the original graph.

    Parameters:
    - original_graph: NetworkX graph object

    Returns:
    - random_graph: Random graph with properties preserved from the existing graph
    """

    # If the existing graph is directed, ensure the generated graph is directed as well
    if original_graph.is_directed():
        in_degree = dict(original_graph.in_degree())
        out_degree = dict(original_graph.out_degree())
        random_graph = nx.directed_configuration_model(
            in_degree.values(), out_degree.values(), seed=seed
        )
    else:
        degree = dict(original_graph.degree())
        random_graph = nx.configuration_model(degree.values(), seed=seed)

    # Adjust any additional properties of the generated graph to match those of the existing graph
    # Update node attributes only for nodes that exist in both graphs
    common_nodes = set(original_graph.nodes()).intersection(random_graph.nodes())
    for node in common_nodes:
        random_graph.nodes[node].update(original_graph.nodes[node])

    # remove self loops
    random_graph.remove_edges_from(nx.selfloop_edges(random_graph))

    # Add edges between separated components until the graph becomes connected
    if original_graph.is_directed():
        while not nx.is_weakly_connected(random_graph):
            # Find the weakly connected components
            components = list(nx.weakly_connected_components(random_graph))

            # Add an edge between a random node in each component
            node1 = list(components[0])[0]
            node2 = list(components[1])[0]
            random_graph.add_edge(node1, node2)
    else:
        while not nx.is_connected(random_graph):
            # Find the connected components
            components = list(nx.connected_components(random_graph))

            # Add an edge between a random node in each component
            node1 = list(components[0])[0]
            node2 = list(components[1])[0]
            random_graph.add_edge(node1, node2)

    return random_graph


# -------------------- MAIN --------------------#

# Directory containing the edge list files
directory = "../data/motifs/"

# List to store the graphs
motifs = []

# Iterate over each file in the directory
for i in range(
    1, 14
):  # Assuming files are named file1.edges, file2.edges, ..., file13.edges
    file_name = f"motif{i}.edges"
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        # Read the graph from the edge list file
        motif = read_graph_from_edge_list(file_path)
        motifs.append(motif)
        print(f"Graph {i} added to the list.")
    else:
        print(f"File {file_name} not found.")


# Read the real-world graph
real_world_graph = read_graph_from_edge_list("../data/twitter/12831.edges")

# Count the occurrences of each motif in the real-world graph
counts = subgraph_count(real_world_graph, motifs)

# Create a dataframe with the counts and save it to a CSV file
counts_df = pd.DataFrame(counts.items(), columns=["Motif", "Count"])
counts_df.to_csv("../data/sheets/motif_counts.csv", index=False)


# Generate random graphs using the configuration model
seed_list = [i for i in range(20)]

random_graphs = [
    generate_configuration_model_graph(real_world_graph, seed_list[i])
    for i in range(20)
]

del real_world_graph

# Count the occurrences of each motif in each random graph
random_graph_counts = [
    subgraph_count(random_graph, motifs) for random_graph in random_graphs
]

del random_graphs

# Create a dataframe with the counts for each random graph and save it to a CSV file
random_counts_df = pd.DataFrame(random_graph_counts)
random_counts_df.columns = [f"Motif {i+1}" for i in range(13)]
random_counts_df.to_csv(
    "../data/sheets/random_counts.csv", index=[f"random_graph_{i+1}" for i in range(10)]
)

del random_counts_df

# Calculate the average of the counts for each motif in the random graphs
average_counts = {
    i: sts.mean([counts[i] for counts in random_graph_counts])
    for i in range(len(motifs))
}

# Calculate the standard deviation of subgraph counts in random graphs
std_dev_counts = {
    i: sts.stdev([counts[i] for counts in random_graph_counts])
    for i in range(len(motifs))
}

# Calculate the Z-score of subgraph counts in real-world graph
z_scores = {}
for i in range(len(motifs)):
    if std_dev_counts[i] != 0:
        z_scores[i] = (counts[i] - average_counts[i]) / std_dev_counts[i]
    else:
        # Handle the case when standard deviation is zero
        z_scores[i] = float("nan")


# generate data frame with average counts, standard deviation and z-scores
motif_df = pd.DataFrame(
    {
        "Motif": list(average_counts.keys()),
        "Average Count": list(average_counts.values()),
        "Standard Deviation": list(std_dev_counts.values()),
        "Z-Score": list(z_scores.values()),
    }
)

motif_df["Motif"] = motif_df["Motif"] + 1
motif_df.index = motif_df["Motif"]
motif_df.drop(columns="Motif", inplace=True)

# Save the dataframe to a CSV file
motif_df.to_csv("../data/sheets/motifs.csv")
