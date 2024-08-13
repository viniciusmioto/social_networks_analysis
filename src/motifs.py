#!./venv/bin/python

# -------------------- ABOUT -------------------- #
#
# Author: Vinícius Mioto
# Professor: André Luís Vignatti
# BSc. Computer Science
# Universidade Federal do Paraná
#
# ----------------------------------------------- #


# -------------------- CONSTANTS -------------------- 

# Directory containing the network files (edge lists)
GRAPH_FILES_DIRECTORY = "../data/gplus/"

# File extension for the network files
FILE_EXTENSION = ".edges"

# Directory containing the edge list files
MOTIFS_PATH = "../data/motifs/"

# Number of random graphs to generate
NUM_RANDOM_GRAPHS = 25

# Directory to save the results
RESULTS_DIRECTORY = "../data/"

# Sampling sizes (percentage of the original graph) | 1 for no sampling
SAMPLING_SIZES = [0.1, 0.2, 0.3, 0.4]

# -------------------- LIBRARIES -------------------- #


import pandas as pd
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statistics as sts
from networkx.algorithms import isomorphism
import itertools
import random

# -------------------- FUNCTIONS -------------------- #


def plot_graph(graph):
    """Plots a directed graph."""
    pos = nx.spring_layout(graph, seed=42)  # Positions for all nodes
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=300,
        node_color="skyblue",
        font_size=10,
        arrowsize=10,
    )
    plt.show()


def get_graph_name(file_path):
    """Extracts the name of a graph from its file path."""
    return os.path.basename(file_path).split(".")[0]


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


def read_graphs_from_directory(directory):
    """
    Reads all graphs from a directory containing edge list files.

    Parameters:
        directory (str): The path to the directory containing the edge list files.

    Returns:
        dict: key: graph name, value: NetworkX graph object.
    """
    graphs = {}
    for file in os.listdir(directory):
        if file.endswith(FILE_EXTENSION):
            graph_name = get_graph_name(file)
            graph = read_graph_from_edge_list(os.path.join(directory, file))
            graphs[graph_name] = graph
    return graphs


def generate_subgraphs(graph, size):
    """
    Generate all subgraphs of a given size from a graph.

    Parameters:
        graph (NetworkX graph): The input graph from which subgraphs are generated.
        size (int): The size of subgraphs to generate (number of nodes).

    Returns:
        list: List of subgraphs of the specified size.
    """
    subgraphs = []
    for nodes in itertools.combinations(graph.nodes(), size):
        # guarantee that there is no isolated node
        subgraph = graph.subgraph(nodes)
        if nx.is_weakly_connected(subgraph):
            subgraphs.append(subgraph)

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
    # Find the largest size of the motifs
    max_size = max([subgraph.number_of_nodes() for subgraph in motifs])
    print("Generating subgraphs of size", max_size)

    # Generate all subgraphs of the largest size in the list of motifs
    all_subgraphs = generate_subgraphs(graph, max_size)
    print("Generated", len(all_subgraphs), "subgraphs")

    # Initialize a dictionary to store counts for each motif
    motif_counts = {i: 0 for i, motif in enumerate(motifs)}

    # Iterate through all subgraphs and motifs to count occurrences
    for subgraph in all_subgraphs:
        for i, motif in enumerate(motifs):
            if len(subgraph.edges()) == len(motif.edges()):
                if nx.is_isomorphic(subgraph, motif):
                    motif_counts[i] += 1

    return motif_counts


def generate_sample_graph(original_graph, sample_size, seed=42):
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

    # Sample nodes using the seed
    random.seed(seed)
    sample_nodes = random.sample(nodes_list, sample_size)

    # Add all sampled nodes to the sample graph
    sample_graph.add_nodes_from(sample_nodes)

    # Add all edges that exists in the original graph to the sample graph
    for edge in original_graph.edges():
        if edge[0] in sample_nodes and edge[1] in sample_nodes:
            sample_graph.add_edge(edge[0], edge[1])

    return sample_graph


def generate_configuration_model_graph(original_graph, seed=42):
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


# -------------------- MAIN -------------------- #

# ---------------- LOAD FILES ------------------ #

# List to store the graphs
motifs = []

# Iterate over each file in the MOTIFS_PATH
for i in range(
    1, 14
):  # Assuming files are named file1.edges, file2.edges, ..., file13.edges
    file_name = f"motif{i}.edges"
    file_path = os.path.join(MOTIFS_PATH, file_name)
    if os.path.exists(file_path):
        # Read the graph from the edge list file
        motif = read_graph_from_edge_list(file_path)
        motifs.append(motif)
        print(f"Motif {i} added to the list.")
    else:
        print(f"File {file_name} not found.")


print("\nMotifs loaded.\n")


# Read all Real-World Graphs from the directory
real_world_graphs = []
graph_names = []

# Add all graphs to the list of NX graphs
for graph_name, graph in read_graphs_from_directory(GRAPH_FILES_DIRECTORY).items():
    real_world_graphs.append(graph)
    graph_names.append(graph_name)
    print(f"Graph {graph_name} added to the list.")

print("\nReal-World Graphs loaded.\n")

# ---------------- ANALYSIS ------------------ #

# Create a dataframe with the graph_name, average_count, standard_deviation, 
# z_score, and significance_profile for each motif and each sample_size
summary_df = pd.DataFrame(
    columns=["graph_name", "motif", "sample_size", "average_count", "standard_deviation", "z_score", "significance_profile"]
)

# Seed list for sample graph generation
seeds_sample_graph = [i for i in range(len(SAMPLING_SIZES))]

# Iterate over each real-world graph
for graph_index, real_world_graph in enumerate(real_world_graphs):
    # Iterate over each sampling size
    for sample_size_index, sample_size in enumerate(SAMPLING_SIZES):
        print(
            f"\nStarting analysis {graph_names[graph_index]} | Graph {graph_index+1}/{len(real_world_graphs)} | Sampling Size {sample_size}\n"
        )

        # Start using only the sample graph for the analysis
        # If the sample size is 1, the sample graph is the original graph

        # Check if the sampling size is 1
        if sample_size == 1:
            print("Working with the original graph.")
            sample_graph = real_world_graph
            graph_name = graph_names[graph_index]
        else:
            # Create a sample graph from the original graph
            sample_graph = generate_sample_graph(
                real_world_graph,
                int(sample_size * real_world_graph.number_of_nodes()),
                seed=seeds_sample_graph[sample_size_index],
            )
            graph_name = graph_names[graph_index] + f"_sample_{sample_size}"

            print(
                f"\nStarting analysis {graph_names[graph_index]} | Graph {graph_index+1}/{len(real_world_graphs)}\n"
            )

        print("Counting motifs in the sample of original graph...")
        # Count the occurrences of each motif in the real-world graph
        counts = subgraph_count(sample_graph, motifs)

        # Create a dataframe with the counts and save it to a CSV file
        # This dataframe has one line, and each column corresponds to a motif
        motif_counts_df = pd.DataFrame(counts, index=["original"])

        # Save the counts to a CSV file
        # motif_counts_df.to_csv("../data/sheets/motif_counts.csv", index=False)

        # Rename the columns to match the motif numbers
        motif_counts_df.columns = [f"motif_{i}" for i in range(1, 14)]

        # Generate random graphs using the configuration model
        seeds_random_graphs = [i for i in range(NUM_RANDOM_GRAPHS)]

        random_graphs = [
            generate_configuration_model_graph(sample_graph, seeds_random_graphs[i])
            for i in range(NUM_RANDOM_GRAPHS)
        ]

        print("Random graphs generated.")

        # Print a message indicating the start of counting motifs in random graphs
        print("Starting to count motifs in random graphs...")

        # Initialize an empty list to store counts for each random graph
        random_graph_counts_all = []

        # Count the occurrences of each motif in each random graph
        for i, random_graph in enumerate(random_graphs):
            print(f"Counting motifs in random graph {i+1}")
            random_graph_counts = subgraph_count(random_graph, motifs)
            random_graph_counts_all.append(random_graph_counts)

        # Create a DataFrame to store the counts for each random graph
        columns = [f"motif_{i+1}" for i in range(len(motifs))]
        index = [f"rand_graph_{i+1}" for i in range(len(random_graphs))]
        random_counts_df = pd.DataFrame(columns=columns, index=index)

        # Fill the DataFrame with the counts for each random graph
        for i, random_graph_counts in enumerate(random_graph_counts_all):
            for j, count in random_graph_counts.items():
                random_counts_df.loc[f"rand_graph_{i+1}", f"motif_{j+1}"] = count

        # Save the DataFrame to a CSV file
        # random_counts_df.to_csv("../data/sheets/random_counts.csv")

        # Print a message indicating that counts for the random graphs are saved to a CSV file
        print("Counts for the random graphs saved to CSV file.")

        print("Calculating average counts, standard deviation, and Z-scores...")

        # Calculate the average of the counts for each motif in the random graphs
        average_counts = random_counts_df.mean()

        # Calculate the standard deviation of the counts for each motif in the random graphs
        std_dev = random_counts_df.std()

        # Calculate the Z-scores for each motif, avoiding division by zero
        z_scores = (motif_counts_df.iloc[0] - average_counts) / np.where(
            std_dev == 0, 1, std_dev
        )

        # Calculate the Network Significance Profile (SP) for each motif
        # Formula: SP_i=Z_i/\sqrt{\sum_jZ_j^2}
        # Avoid division by zero by normalizing the Z-scores
        significance_profile = (
            z_scores / np.sqrt(np.sum(z_scores**2)) if np.sum(z_scores**2) != 0 else 0.0
        )

        # Add the results to the summary DataFrame
        # Use a auxiliary DataFrame to combine the results
        aux_df = pd.DataFrame(
            {
                "graph_name": graph_name,
                "motif": [f"motif_{i+1}" for i in range(len(motifs))],
                "sample_size": sample_size,
                "average_count": average_counts.values,
                "standard_deviation": std_dev.values,
                "z_score": z_scores.values,
                "significance_profile": significance_profile,
            }
        )

        summary_df = pd.concat([summary_df, aux_df], ignore_index=True)

        print("\nResults of graph", graph_names[graph_index], "saved to the summary DataFrame.\n")

# Save the summary DataFrame to a CSV file
summary_df.to_csv(os.path.join(RESULTS_DIRECTORY, "summary.csv"), index=False)

print("Summary saved to CSV file.\n\n")