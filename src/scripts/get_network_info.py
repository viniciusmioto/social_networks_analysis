import sys
import os

# Get the absolute path to the src directory
src_path = os.path.abspath(os.path.join(os.getcwd(), "../"))

# Add src_path to the system path
sys.path.insert(0, src_path)

# Now you can import your module
import utils.graph_utils as gru

import pandas as pd

GRAPH_FILES_DIRECTORY = "../../data/enzymes/"
FILE_EXTENSION = ".edges"
OUTPUT_FILE = "../../results/enzymes_info.csv"


def main():
    graph_info_df = pd.DataFrame(
        columns=["graph", "nodes", "edges", "max_degree", "avg_degree"]
    )

    graphs = gru.read_graphs_from_directory(GRAPH_FILES_DIRECTORY, FILE_EXTENSION)

    for graph_name, graph in graphs.items():
        info = gru.get_graph_info(graph)
        print(f"Graph: {graph_name}")

        # save info to the dataframe
        graph_info_df = graph_info_df._append(
            {
                "graph": graph_name,
                "nodes": info["nodes"],
                "edges": info["edges"],
                "max_degree": info["max_degree"],
                "avg_degree": info["avg_degree"],
            },
            ignore_index=True,
        )

    # save the results to a csv file
    graph_info_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
