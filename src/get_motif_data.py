#!./venv/bin/python

# Get the following data for each graph:
# graph_name, 
# motif,
# sample_size, 
# average_count, 
# standard_deviation, 
# z_score,significance_profile

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
GRAPH_FILES_DIRECTORY = "../data/enzymes/"

# File extension for the network files
FILE_EXTENSION = ".edges"

# Directory containing the edge list files
MOTIFS_PATH = "../data/motifs/"

# Number of random graphs to generate
NUM_RANDOM_GRAPHS = 25

# Directory to save the results
RESULTS_DIRECTORY = "../data/"

# Sampling sizes (percentage of the original graph) | 1 for no sampling
SAMPLING_SIZES = [1, 0.1, 0.2, 0.3, 0.4]

# -------------------- LIBRARIES -------------------- #

import graph_utils as gu
import os
import pandas as pd
import numpy as np


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
        motif = gu.read_graph_from_edge_list(file_path)
        motifs.append(motif)
        print(f"Motif {i} added to the list.")
    else:
        print(f"File {file_name} not found.")


print("\nMotifs loaded.\n")


# Read all Real-World Graphs from the directory
real_world_graphs = []
graph_names = []

# Add all graphs to the list of NX graphs
for graph_name, graph in gu.read_graphs_from_directory(GRAPH_FILES_DIRECTORY).items():
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
            sample_graph = gu.generate_sample_graph(
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
        counts = gu.subgraph_count(sample_graph, motifs)

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
            gu.generate_configuration_model_graph(sample_graph, seeds_random_graphs[i])
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
            random_graph_counts = gu.subgraph_count(random_graph, motifs)
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