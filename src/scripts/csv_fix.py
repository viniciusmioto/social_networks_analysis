# -------------------- CONSTANTS --------------------

# Directory containing the network files (edge lists)
GRAPH_FILES_DIRECTORY = "../../data/twitter_samples/"

# File extension for the network files
FILE_EXTENSION = ".edges"

# Directory containing the edge list files
MOTIFS_PATH = "../../data/motifs/"

# Number of random graphs to generate
NUM_RANDOM_GRAPHS = 25

# Directory to save the results
RESULTS_DIRECTORY = "../../data/"


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
    1, 14
):  # Assuming files are named file1.edges, file2.edges, ..., file13.edges
    file_name = f"motif{i}.edges"
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
csv_names = []

# Add all graphs to the list of NX graphs
for graph_name, graph in gru.read_graphs_from_directory(GRAPH_FILES_DIRECTORY).items():
    real_world_graphs.append(graph)
    graph_names.append(graph_name)
    print(f"Graph {graph_name} added to the list.")
    
    for i in range(1, 14):
        csv_names.append(graph_name)


print("\nReal-World Graphs loaded.\n")

# ---------------- ANALYSIS ------------------ #

# Create a data-frame with the graph_name, average_count, standard_deviation,
# z_score, and significance_profile for each motif
summary_df = pd.DataFrame(csv_names)


# Save the summary DataFrame to a CSV file
summary_df.to_csv(os.path.join(RESULTS_DIRECTORY, "summary.csv"), index=False)

print("Summary saved to CSV file.\n\n")
