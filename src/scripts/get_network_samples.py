import sys
import os
import networkx as nx
import graph_utils as gru

# Directory containing the network files (edge lists)
GRAPH_FILES_DIRECTORY = "../../data/twitter/"

# Directory to save the results
RESULTS_DIRECTORY = "../../data/twitter_samples/"

# File extension for the network files
FILE_EXTENSION = ".edges"

# Sampling sizes (percentage of the original graph) | 1 for no sampling
SAMPLING_SIZES = [0.10, 0.15, 0.20]

SAMPLING_METHODS = ["rn", "rpn", "rw", "sff"]

# Get the absolute path to the src directory
src_path = os.path.abspath(os.path.join(os.getcwd(), "../"))

# Add src_path to the system path
sys.path.insert(0, src_path)


# ---------------- LOAD REAL-WORLD GRAPHS ------------------ #

# Read all Real-World Graphs from the directory
# These are the graphs to be analyzed
real_world_graphs = []
graph_names = []
sample_graphs = []

# Add all graphs to the list of NX graphs
for graph_name, graph in gru.read_graphs_from_directory(GRAPH_FILES_DIRECTORY).items():
    real_world_graphs.append(graph)
    graph_names.append(graph_name)
    print(f"Graph {graph_name} added to the list.")

print("\nReal-World Graphs loaded.\n")

# Seed list for sample graph generation
# the first sample_percent graph will have seed 0,
# the second will have seed 1, and so on
seeds_sample_graph = [i for i in range(len(SAMPLING_SIZES))]
print("Seed list for sample graph generation created:")
print(seeds_sample_graph)

# Iterate over each real-world graph
for graph_index, real_world_graph in enumerate(real_world_graphs):
    # Iterate over each sampling size
    for sample_method_index, sample_method in enumerate(SAMPLING_METHODS):
        # Iterate over each sampling method
        for sample_size_index, sample_percent in enumerate(SAMPLING_SIZES):
            print(f"\nGenerating Sample {graph_names[graph_index]}")
            print(f" --> Graph {graph_index+1}/{len(real_world_graphs)}")
            print(f" --> Sampling Size {sample_percent}")
            print(f" --> Sampling Method {sample_method}\n")

            # Check if the sampling size is 1 --> no sampling
            if sample_percent == 1:
                print("Working with the original graph.")
                sample_graph = real_world_graph
                graph_name = graph_names[graph_index]
            else:
                # Create a sample graph from the original graph
                sample_graph = gru.get_sample_by_method(
                    real_world_graph,
                    sample_percent,
                    method=SAMPLING_METHODS[sample_method_index],
                    seed=seeds_sample_graph[sample_size_index],
                )
                graph_name = (
                    graph_names[graph_index]
                    + f"_sample_{sample_method}_{int(sample_percent * 100)}"
                )

            # Save graph files
            nx.write_edgelist(
                sample_graph, path=RESULTS_DIRECTORY + graph_name + FILE_EXTENSION,
                data=False
            )
