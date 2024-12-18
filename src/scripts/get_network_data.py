# -------------------- CONSTANTS --------------------

# Directory containing the network files (edge lists)
GRAPH_FILES_DIRECTORY = "../../data/twitter_samples/"

# File extension for the network files
FILE_EXTENSION = ".edges"

# name before the index (e.g. triangle1.edges), 
# let empty for index only
MOTIFS_NAME = 'claw'

NUM_MOTIFS = 9

# Directory containing the edge list files
MOTIFS_PATH = "../../data/motifs_4/"

# Matching method
IS_ANCHORED = True

# Number of random graphs to generate
NUM_RANDOM_GRAPHS = 25

# Directory to save the results
RESULTS_DIRECTORY = "../../data/"

RESULT_FILE = "twitter_motifs_4_anchored.csv"



# -------------------- LIBRARIES -------------------- #

import sys
import os
import pandas as pd
import numpy as np

# Get the absolute path to the src directory
src_path = os.path.abspath(os.path.join(os.getcwd(), "../"))

# Add src_path to the system path
sys.path.insert(0, src_path)

# Now you can import your module
import scripts.graph_utils as gru

# ---------------- LOAD SELECTED MOTIFS ------------------ #

# List to store the selected motifs (13 motifs, all with 3 nodes)
motifs = []

# Iterate over each file in the MOTIFS_PATH
for i in range(
    1, NUM_MOTIFS + 1
):  # Assuming files are named file1.edges, file2.edges, ..., file13.edges
    file_name = f"{MOTIFS_NAME}{i}.edges"
    file_path = os.path.join(MOTIFS_PATH, file_name)
    if os.path.exists(file_path):
        # Read the graph from the edge list file
        motif = gru.read_directed_graph_from_edge_list(file_path)
        motifs.append(motif)
        print(f"Motif {i} added to the list.")
    else:
        print(f"File {file_name} not found.")


print("\nThe selected Motifs were loaded.\n")


# ---------------- LOAD REAL-WORLD GRAPHS ------------------ #

# Read all Real-World Graphs from the directory
# These are the graphs to be analyzed
real_world_graphs = []
graph_names = []

# Add all graphs to the list of NX graphs
for graph_name, graph in gru.read_graphs_from_directory(GRAPH_FILES_DIRECTORY).items():
    real_world_graphs.append(graph)
    graph_names.append(graph_name)
    print(f"Graph {graph_name} added to the list.")

print("\nReal-World Graphs loaded.\n")

# ---------------- ANALYSIS ------------------ #

# Create a data-frame with the graph_name, average_count, standard_deviation,
# z_score, and significance_profile for each motif
summary_df = pd.DataFrame(
    columns=[
        "graph_name",
        "motif",
        "average_count",
        "standard_deviation",
        "z_score",
        "significance_profile",
    ]
)


# Iterate over each real-world graph
for graph_index, real_world_graph in enumerate(real_world_graphs):
    # Count the occurrences of each motif in the real-world graph
    counts = gru.subgraph_count(real_world_graph, motifs, anchored=IS_ANCHORED)

    # Create a data-frame with the counts
    # This data-frame has one line, and each column corresponds to a motif
    motif_counts_df = pd.DataFrame(counts, index=["original"])

    # Rename the columns to match the motif numbers
    motif_counts_df.columns = [f"motif_{i}" for i in range(1, NUM_MOTIFS + 1)]

    # Generate random graphs using the configuration model
    seeds_random_graphs = [i for i in range(NUM_RANDOM_GRAPHS)]

    # the first random graph will have seed 0,
    # the second will have seed 1, and so on
    random_graphs = [
        gru.generate_configuration_model_graph(real_world_graph, seeds_random_graphs[i])
        for i in range(NUM_RANDOM_GRAPHS)
    ]

    print("Random graphs generated. Starting to count motifs in random graphs...")

    # Initialize an empty list to store counts for each random graph
    random_graph_counts_all = []

    # Count the occurrences of each motif in each random graph
    for i, random_graph in enumerate(random_graphs):
        print(f"Counting motifs in random graph {i+1}")
        random_graph_counts = gru.subgraph_count(random_graph, motifs, anchored=IS_ANCHORED)
        random_graph_counts_all.append(random_graph_counts)

    # Create a DataFrame to store the counts for each random graph
    columns = [f"motif_{i+1}" for i in range(len(motifs))]
    index = [f"rand_graph_{i+1}" for i in range(len(random_graphs))]
    random_counts_df = pd.DataFrame(columns=columns, index=index)

    # Fill the DataFrame with the counts for each random graph
    for i, random_graph_counts in enumerate(random_graph_counts_all):
        for j, count in random_graph_counts.items():
            random_counts_df.loc[f"rand_graph_{i+1}", f"motif_{j+1}"] = count

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

    info = gru.get_graph_info(real_world_graph)

    # Add the results to the summary DataFrame
    # Use a auxiliary DataFrame to combine the results
    aux_df = pd.DataFrame(
        {
            "graph_name": graph_names[graph_index],
            "motif": [f"{MOTIFS_NAME}{i+1}" for i in range(len(motifs))],
            "average_count": average_counts.values,
            "standard_deviation": std_dev.values,
            "z_score": z_scores.values,
            "significance_profile": significance_profile,
            "nodes": info["nodes"],
            "edges": info["edges"],
            "max_degree": info["max_degree"],
            "avg_degree": info["avg_degree"],
        }
    )

    summary_df = pd.concat([summary_df, aux_df], ignore_index=True)

    print(
        "\nResults of graph",
        graph_names[graph_index],
        "saved to the summary DataFrame.\n",
    )

# Save the summary DataFrame to a CSV file
summary_df.to_csv(os.path.join(RESULTS_DIRECTORY + RESULT_FILE), index=False)

print("Summary saved to CSV file.\n\n")
