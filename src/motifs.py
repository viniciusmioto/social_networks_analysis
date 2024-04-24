import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statistics as sts
from networkx.algorithms import isomorphism
import itertools
import random


def read_graph_from_edge_list(file_path):
    """Reads a directed graph from an edge list file."""
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
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
        subgraphs.append(graph.subgraph(nodes))

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
            if nx.is_isomorphic(subgraph, motif):
                motif_counts[i] += 1

    return motif_counts


def generate_configuration_model_graph(original_graph, seed):
    """
    Generate a random graph using the configuration model while preserving the properties of an existing graph.

    Parameters:
    - original_graph: NetworkX graph object

    Returns:
    - random_graph: Random graph with properties preserved from the existing graph
    """

    # Step 3: Determine the degree sequence of the existing graph
    degree_sequence = [d for n, d in original_graph.degree()]

    # Step 4: Generate a random graph using the configuration model with the specified degree sequence
    random_graph = nx.configuration_model(degree_sequence, seed=seed)

    # Step 5: If the existing graph is directed, ensure the generated graph is directed as well
    if original_graph.is_directed():
        random_graph = random_graph.to_directed()

    # Step 6: Adjust any additional properties of the generated graph to match those of the existing graph
    # Update node attributes only for nodes that exist in both graphs
    common_nodes = set(original_graph.nodes()).intersection(random_graph.nodes())
    for node in common_nodes:
        random_graph.nodes[node].update(original_graph.nodes[node])

    return random_graph


# Directory containing the edge list files
directory = '../data/motifs/'

# List to store the graphs
motifs = []

# Iterate over each file in the directory
for i in range(1, 14):  # Assuming files are named file1.edges, file2.edges, ..., file13.edges
    file_name = f"motif{i}.edges"
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        # Read the graph from the edge list file
        motif = read_graph_from_edge_list(file_path)
        motifs.append(motif)
        print(f"Graph {i} added to the list.")
    else:
        print(f"File {file_name} not found.")



# read real-world graph
twitter_graph = read_graph_from_edge_list('../data/twitter/14630490.edges')
print("Real-World graph loaded")

# count the subgraphs in the real-world graph
print("Counting subgraphs in the real-world graph")
counts = subgraph_count(twitter_graph, motifs)


# show the subgraph counts in a pandas dataframe
twitter_motifs_df = pd.DataFrame(counts.items(), columns=['Motif', 'Count'])
twitter_motifs_df['Motif'] = twitter_motifs_df['Motif'] + 1

# Generate the seed list for the configuration model
seed_list = [i for i in range(twitter_graph.number_of_nodes())]


print("Generating random graphs")
# With the same degree sequence as the Twitter graph
random_graphs = [generate_configuration_model_graph(twitter_graph, seed_list[i]) 
                 for i in range(10)]

# Calculate the counts of subgraphs in each random graph
random_graph_counts = [subgraph_count(graph, motifs) for graph in random_graphs]

# show the counts of subgraphs in each random graph in a pandas dataframe 
random_graphs_df = pd.DataFrame(random_graph_counts)

# rename the columns to the motif numbers
random_graphs_df.columns = [f"motif{i+1}" for i in range(13)]

# calculate the average of subgraph counts in random graphs
average_counts = {
    i: sts.mean([counts[i] for counts in random_graph_counts])
    for i in range(len(motifs))
}

# calculate the standard deviation of subgraph counts in random graphs
std_dev_counts = {
    i: sts.stdev([counts[i] for counts in random_graph_counts])
    for i in range(len(motifs))
}

# calculate the Z-score of subgraph counts in real-world graph
z_scores = {}
for i in range(len(motifs)):
    if std_dev_counts[i] != 0:
        z_scores[i] = (counts[i] - average_counts[i]) / std_dev_counts[i]
    else:
        # Handle the case when standard deviation is zero
        z_scores[i] = float('nan')


# generate data frame with average counts, standard deviation and z-scores
motifs_df = pd.DataFrame({
    'Motif': list(average_counts.keys()),
    'Average Count': list(average_counts.values()),
    'Standard Deviation': list(std_dev_counts.values()),
    'Z-Score': list(z_scores.values())
})

# export the motifs_df to a csv file
motifs_df.to_csv('../data/motifs/motifs.csv', index=False)
