import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
import random
from collections import deque
from PIL import Image


# -------------------- FUNCTIONS -------------------- #


def get_graph_info(graph):
    """
    Get information about the graph
    """
    graph_degree = dict(graph.degree())

    info = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "max_degree": max(graph_degree.values()) / 2,
        "avg_degree": sum(graph_degree.values()) / len(graph_degree),
    }
    return info


def draw_graph(graph, seed=42):
    """Plots a directed graph."""
    pos = nx.spring_layout(graph, seed=seed)  # Positions for all nodes
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


def get_image_paths_from_directory(
    directory, valid_extensions=(".png", ".jpg", ".jpeg")
):
    """
    Scans a directory and returns a list of image file paths.
    Filters files by valid image extensions.
    """
    return sorted(
        [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.lower().endswith(valid_extensions)
        ]
    )


def plot_line_chart_with_images(
    data, title, x_label, y_label, image_directory, image_y_offset=-0.07
):
    # Get image paths from the directory
    image_paths = get_image_paths_from_directory(image_directory)

    # Ensure that the number of images matches the number of motifs
    motifs = data["motif"].unique()
    if len(image_paths) != len(motifs):
        raise ValueError(
            f"Number of images ({len(image_paths)}) does not match the number of motifs ({len(motifs)})."
        )

    fig = go.Figure()

    # Add the line chart
    for graph_name in data["graph_name"].unique():
        df = data[data["graph_name"] == graph_name]
        fig.add_trace(
            go.Scatter(
                x=df["motif"], y=df[y_label], mode="lines+markers", name=graph_name
            )
        )

    # Hide the default X-axis tick labels (motif_1, motif_2, etc.)
    fig.update_xaxes(showticklabels=True)

    # Add images as custom X-tick labels
    for i, image_path in enumerate(image_paths):
        fig.add_layout_image(
            dict(
                source=Image.open(image_path),  # Open image from path
                xref="x",
                yref="paper",  # Use 'paper' for relative y positioning
                x=motifs[i],  # Place at each x tick
                y=image_y_offset,  # Adjust below the x-axis
                sizex=0.5,  # Adjust size
                sizey=0.5,
                xanchor="center",
                yanchor="top",
            )
        )

    # Adjust layout and display
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_label,
        xaxis=dict(tickvals=motifs),  # Ensure correct x-axis positions for the motifs
    )
    fig.show()


# Plot line chart of Z-scores for each graph_name
def plot_line_chart(data, title, x_label, y_label):
    fig = go.Figure()
    for graph_name in data["graph_name"].unique():
        df = data[data["graph_name"] == graph_name]
        fig.add_trace(
            go.Scatter(
                x=df["motif"], y=df[y_label], mode="lines+markers", name=graph_name
            )
        )
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.show()


def plot_degree_distribution(G):
    # In-degree and out-degree for directed graphs
    if G.is_directed():
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

        in_degree_values = list(in_degrees.values())
        out_degree_values = list(out_degrees.values())

        # In-degree distribution plot
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(
            in_degree_values,
            bins=range(min(in_degree_values), max(in_degree_values) + 1, 1),
            alpha=0.75,
            color="blue",
            edgecolor="black",
        )
        plt.title("In-Degree Distribution")
        plt.xlabel("In-Degree")
        plt.ylabel("Frequency")

        # Out-degree distribution plot
        plt.subplot(1, 2, 2)
        plt.hist(
            out_degree_values,
            bins=range(min(out_degree_values), max(out_degree_values) + 1, 1),
            alpha=0.75,
            color="green",
            edgecolor="black",
        )
        plt.title("Out-Degree Distribution")
        plt.xlabel("Out-Degree")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()
    else:
        print("The graph is not directed. Use a directed graph (DiGraph).")


def plot_degree_distribution_scatter(G, log_log=True):
    # In-degree and out-degree for directed graphs
    if G.is_directed():
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

        in_degree_values = list(in_degrees.values())
        out_degree_values = list(out_degrees.values())

        plt.figure(figsize=(12, 6))

        # In-degree scatter plot
        plt.subplot(1, 2, 1)
        in_deg_count = np.bincount(in_degree_values)
        plt.scatter(range(len(in_deg_count)), in_deg_count, color="blue")
        plt.title("In-Degree Distribution")
        plt.xlabel("In-Degree")
        plt.ylabel("Frequency")

        if log_log:
            plt.xscale("log")
            plt.yscale("log")

        # Out-degree scatter plot
        plt.subplot(1, 2, 2)
        out_deg_count = np.bincount(out_degree_values)
        plt.scatter(range(len(out_deg_count)), out_deg_count, color="green")
        plt.title("Out-Degree Distribution")
        plt.xlabel("Out-Degree")
        plt.ylabel("Frequency")

        if log_log:
            plt.xscale("log")
            plt.yscale("log")

        plt.tight_layout()
        plt.show()
    else:
        print("The graph is not directed. Use a directed graph (DiGraph).")


def get_graph_name(file_path):
    """Extracts the name of a graph from its file path."""
    return os.path.basename(file_path).split(".")[0]


def read_directed_graph_from_edge_list(file_path):
    """Reads a directed graph from an edge list file."""
    G = nx.DiGraph()
    with open(file_path, "r") as file:
        for line in file:
            edge = line.strip().split()
            if len(edge) == 2:
                source, target = edge
                G.add_edge(source, target)
    return G


def read_graphs_from_directory(directory, file_extension=".edges"):
    """
    Reads all graphs from a directory containing edge list files.

    Parameters:
        directory (str): The path to the directory containing the edge list files.

    Returns:
        dict: key: graph name, value: NetworkX graph object.
    """
    graphs = {}
    for file in os.listdir(directory):
        if file.endswith(file_extension):
            graph_name = get_graph_name(file)
            graph = read_directed_graph_from_edge_list(os.path.join(directory, file))
            graphs[graph_name] = graph
    return graphs


def generate_subgraphs(graph, size):
    """
    Generate all subgraphs of a given size from a graph.
    It uses combination (num_nodes, size) to generate all
    possible subgraphs of the given size.

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


def generate_anchored_subgraphs(graph, size):
    """
    Generate all subgraphs of a given size that include a specific anchor node.
    
    Parameters:
        graph (NetworkX graph): The input graph.
        anchor (node): The node that should serve as the anchor in each subgraph.
        size (int): The size of subgraphs to generate.
    
    Returns:
        list: List of subgraphs of the specified size that include the anchor node.
    """
    subgraphs = set()

    for node in graph.nodes():
        neighbors = sorted(list(graph.neighbors(node)))
        subgraph_nodes = set(neighbors)
        subgraph_nodes.add(node)

        if len(subgraph_nodes) >= size:
            subgraphs.add(graph.subgraph(subgraph_nodes))
    
    return list(subgraphs)


def subgraph_count(graph, motifs, anchored=False):
    """
    Count the occurrences of a list of subgraphs within a given graph.

    Parameters:
        graph (NetworkX graph): The input graph in which occurrences are counted.
        motifs (list of NetworkX graphs): List of subgraphs whose occurrences are being counted.

    Returns:
        dict: A dictionary containing the counts of occurrences for each motif.
    """
    # Find the largest size of the motifs
    max_size = max([subgraph.number_of_nodes() for subgraph in motifs])
    print("Generating subgraphs of size ", max_size)

    # Generate all subgraphs of the largest size in the list of motifs
    all_subgraphs = generate_anchored_subgraphs(graph, max_size) if anchored else generate_subgraphs(graph, max_size)
    print("Generated", len(all_subgraphs), "subgraphs")

    # Initialize a dictionary to store counts for each motif, starting from 1
    motif_counts = {i: 0 for i in range(len(motifs))}

    # Iterate through all subgraphs and motifs to count occurrences
    for subgraph in all_subgraphs:
        for i, motif in enumerate(motifs):
            if len(subgraph.edges()) == len(motif.edges()):
                # VF2++ Algorithm
                if nx.vf2pp_is_isomorphic(subgraph, motif):
                    motif_counts[i] += 1

    return motif_counts






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


def get_sample_rn(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Node (RN): Selects a random sample of nodes from the input graph
    and includes all edges between the sampled nodes.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())

    # Sample nodes using the seed
    random.seed(seed)
    sample_nodes = random.sample(nodes_list, int(sample_percent * len(nodes_list)))

    # Add all sampled nodes to the sample graph
    sample = original_graph.subgraph(sample_nodes)

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_rdn(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Degree Node (RDN): Selects a random sample of nodes from the input graph
    and includes all edges between the sampled nodes. Each node has a
    probability of getting selected that is proportional to its degree.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())

    # Sample nodes using the seed
    random.seed(seed)

    nodes_probabilities = [i[1] for i in original_graph.degree(nodes_list)]
    sample_nodes = random.choices(
        nodes_list, nodes_probabilities, k=int(len(nodes_list) * sample_percent)
    )

    # Add all sampled nodes to the sample graph
    sample = original_graph.subgraph(sample_nodes)

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_rpn(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Page Rank Node (RDN): Selects a random sample of nodes from the
    input graph and includes all edges between the sampled nodes. Each node has
    a probability of getting selected that is proportional to its page-rank.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())

    # Sample nodes using the seed
    random.seed(seed)

    nodes_probabilities = list(nx.pagerank(original_graph).values())
    sample_nodes = random.choices(
        nodes_list, nodes_probabilities, k=int(len(nodes_list) * sample_percent)
    )

    # Add all sampled nodes to the sample graph
    sample = original_graph.subgraph(sample_nodes)

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_re(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Edge (RE): Selects a random sample of edges from the input graph.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample_graph = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    edges_list = list(original_graph.edges())

    # Sample nodes using the seed
    random.seed(seed)

    sample_edges = random.sample(edges_list, int(len(edges_list) * sample_percent))

    # Add all sampled edges to the sample graph
    sample_graph.add_edges_from(sample_edges)

    return sample_graph


def get_sample_rne(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Node-Edge (RNE): Selects a random node and a random incident edge.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())

    # Sample nodes using the seed
    random.seed(seed)

    sampled_origin_nodes = random.choices(
        nodes_list, k=int(len(nodes_list) * sample_percent)
    )
    sampled_nodes = set(sampled_origin_nodes)

    for s in sampled_origin_nodes:
        neighbors = list(original_graph.neighbors(s))
        if neighbors:
            selected_neighbor = random.sample(list(original_graph.neighbors(s)), 1)[0]
            sampled_nodes.add(selected_neighbor)

    # Add all sampled nodes to the sample graph
    sample = original_graph.subgraph(list(sampled_nodes))

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    return sample_graph


def get_sample_hyb(original_graph, sample_percent, P=0.5, seed=42):
    """
    Generates a hybrid random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of edges from the original_graph.
    Hybrid (HYB): Combines Random Node-Edge (RNE) and Random Edge (RE)
    with probability P and (1-P), respectively.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (float): The percentage of edges to sample.
        P (float): The probability of using RNE. RE is used with probability (1-P).
        seed (int): Seed for random sampling.

    Returns:
        nx.Graph(): NetworkX hybrid sample of the input graph.
    """

    # Directed or Undirected graph
    sample_graph = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    edges_list = list(original_graph.edges())
    nodes_list = list(original_graph.nodes())

    # Set random seed
    random.seed(seed)

    while len(sample_graph.edges()) < int(len(edges_list) * sample_percent):
        # Decide whether to use RNE or RE based on probability P
        if random.random() < P:
            # Use Random Node-Edge (RNE)
            node = random.choice(nodes_list)
            neighbors = list(original_graph.neighbors(node))
            if neighbors:
                neighbor = random.choice(neighbors)
                sample_graph.add_edge(node, neighbor)
        else:
            # Use Random Edge (RE)
            edge = random.sample(edges_list, 1)[0]
            sample_graph.add_edges_from([edge])

    return sample_graph


def get_sample_rnn(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Node Neighbor (RNN): Selects a random node and all incident edges.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())

    # Sample nodes using the seed
    random.seed(seed)

    while len(sample.nodes()) < int(len(nodes_list) * sample_percent):
        sampled_node = random.choice(nodes_list)
        neighbors = list(original_graph.neighbors(sampled_node))

        for neighbor in neighbors:
            sample.add_edge(sampled_node, neighbor)

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_bsf(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Breadth-Search-First (BSF): Selects a random node and all its neighbors.
    Them all the incident edges of the neighbors, and so on.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())
    random.seed(seed)

    # Calculate the number of nodes to sample
    sample_size = int(len(nodes_list) * sample_percent)

    # Set to keep track of visited nodes
    visited = set()

    while len(visited) < sample_size:
        # Pick a random unvisited node if we need more nodes
        unvisited_nodes = list(set(nodes_list) - visited)
        if not unvisited_nodes:
            break

        start_node = random.choice(unvisited_nodes)

        # BFS queue
        queue = deque([start_node])
        visited.add(start_node)

        while queue and len(visited) < sample_size:
            current_node = queue.popleft()

            # Get neighbors of the current node
            for neighbor in original_graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

                    # Stop if the sample has reached the target size
                    if len(visited) >= sample_size:
                        break

    # Create the sample graph from the visited nodes
    sample = original_graph.subgraph(list(visited))

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_dsf(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Depth-First-Search (DFS): Selects a random node and explores as deep as
    possible along each branch before backtracking.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())
    random.seed(seed)

    # Calculate the number of nodes to sample
    sample_size = int(len(nodes_list) * sample_percent)

    # Set to keep track of visited nodes
    visited = set()

    while len(visited) < sample_size:
        # Pick a random unvisited node if we need more nodes
        unvisited_nodes = list(set(nodes_list) - visited)
        if not unvisited_nodes:
            break

        start_node = random.choice(unvisited_nodes)

        # DFS stack
        stack = [start_node]
        visited.add(start_node)

        while stack and len(visited) < sample_size:
            current_node = stack.pop()

            # Get neighbors of the current node
            for neighbor in original_graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

                    # Stop if the sample has reached the target size
                    if len(visited) >= sample_size:
                        break

    # Create the sample graph from the visited nodes
    sample = original_graph.subgraph(list(visited))

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_rw(
    original_graph, sample_percent, fly_back_prob=0.25, max_steps=100, seed=42
):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Walk (RW): Selects a random node and performs a random walk.
    The walker has a chance to "fly back" to a randomly chosen node or pick a
    new starting node if the maximum number of steps is reached.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.
        fly_back_prob (float): The probability of flying back to a random node.
        max_steps (int): The maximum number of steps before picking a new node.
        seed (int): Random seed for reproducibility.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    random.seed(seed)

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())
    sample_size = int(len(nodes_list) * sample_percent)

    # Set to keep track of visited nodes
    visited = set()

    # Start the sampling process
    while len(visited) < sample_size:
        # Pick a random unvisited node if we need more nodes
        unvisited_nodes = list(set(nodes_list) - visited)
        if not unvisited_nodes:
            break

        current_node = random.choice(unvisited_nodes)
        visited.add(current_node)

        step_count = 0

        while len(visited) < sample_size:
            # Random fly-back probability or exceed max steps
            if random.random() < fly_back_prob or step_count >= max_steps:
                # Pick a new random starting node
                unvisited_nodes = list(set(nodes_list) - visited)
                if not unvisited_nodes:
                    break
                current_node = random.choice(unvisited_nodes)
                visited.add(current_node)
                step_count = 0  # Reset the step count after flying back
            else:
                # Choose a random neighbor of the current node
                neighbors = list(original_graph.neighbors(current_node))
                if neighbors:
                    current_node = random.choice(neighbors)
                    if current_node not in visited:
                        visited.add(current_node)
                step_count += 1

    # Create the sample graph from the visited nodes
    sample = original_graph.subgraph(list(visited))

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_rj(original_graph, sample_percent, jump_prob=0.15, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Jump (RJ): Selects a random node and performs a random walk,
    but at each step, the walker has a chance to "jump" to a new random node.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The percentage of nodes to sample.
        jump_prob (float): The probability of jumping to a random new node.
        seed (int): Random seed for reproducibility.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    random.seed(seed)

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())
    sample_size = int(len(nodes_list) * sample_percent)

    # Set to keep track of visited nodes
    visited = set()

    # Start the sampling process
    while len(visited) < sample_size:
        # Pick a random unvisited node if we need more nodes
        unvisited_nodes = list(set(nodes_list) - visited)
        if not unvisited_nodes:
            break

        current_node = random.choice(unvisited_nodes)
        visited.add(current_node)

        while len(visited) < sample_size:
            # Random jump to a new starting node with 'jump_prob'
            if random.random() < jump_prob:
                # Pick a new random unvisited node
                unvisited_nodes = list(set(nodes_list) - visited)
                if not unvisited_nodes:
                    break
                current_node = random.choice(unvisited_nodes)
                visited.add(current_node)
            else:
                # Choose a random neighbor of the current node
                neighbors = list(original_graph.neighbors(current_node))
                if neighbors:
                    current_node = random.choice(neighbors)
                    if current_node not in visited:
                        visited.add(current_node)

    # Create the sample graph from the visited nodes
    sample = original_graph.subgraph(list(visited))

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_drn(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Deletion of Random Node (DRN): Selects a random sample of nodes and delete
    them from the input graph.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample = original_graph.copy()

    nodes_list = list(original_graph.nodes())

    # Sample nodes using the seed
    random.seed(seed)
    sample_nodes = random.sample(nodes_list, int(sample_percent * len(nodes_list)))

    # Add all "unsampled" nodes to the sample graph
    remaining_nodes = [n for n in nodes_list if n not in sample_nodes]

    sample = original_graph.subgraph(remaining_nodes)

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample)))

    return sample_graph


def get_sample_dre(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Random Edge (RE): Selects a random sample of edges from the input graph.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample_graph = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    edges_list = list(original_graph.edges())

    # Sample edges using the seed
    random.seed(seed)

    sample_edges = random.sample(edges_list, int(len(edges_list) * sample_percent))

    remaining_edges = [e for e in edges_list if e not in sample_edges]

    # Add all "unsampled" edges to the sample graph
    sample_graph.add_edges_from(remaining_edges)

    return sample_graph


def get_sample_drne(original_graph, sample_percent, seed=42):
    """
    Generates a random sample of original_graph with sample_percent (%).
    The sample_percent is the percentage of nodes from the original_graph.
    Delete Random Node-Edge (DRNE): Selects a random node and a
    random incident edge, them delete from the graph.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample.

    Returns:
        nx.Graph(): NetworkX random sample of the input graph.
    """

    # Directed or Undirected graph
    sample_graph = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    nodes_list = list(original_graph.nodes())
    edges_list = list(original_graph.edges())

    # Sample nodes using the seed
    random.seed(seed)

    sampled_origin_nodes = random.choices(
        nodes_list, k=int(len(nodes_list) * sample_percent)
    )

    sampled_nodes = set(sampled_origin_nodes)
    sampled_edges = []

    for s in sampled_origin_nodes:
        neighbors = list(original_graph.neighbors(s))
        if neighbors:
            selected_neighbor = random.sample(list(original_graph.neighbors(s)), 1)[0]
            sampled_nodes.add(selected_neighbor)
            sampled_edges.append((s, selected_neighbor))

    # Add all "unsampled" nodes to the sample graph
    remaining_edges = [e for e in edges_list if e not in sampled_edges]

    sample_graph.add_edges_from(remaining_edges)

    return sample_graph


def get_sample_ff(original_graph, sample_percent, p=0.5, r=0.3, seed=42):
    """
    Implements the Forest Fire Sampling (FFS) method.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample as a percentage.
        p (float): Forward burning probability (0 < p < 1).
        r (float): Backward burning ratio (0 < r < 1).
        seed (int): Random seed for reproducibility.

    Returns:
        NetworkX graph: Sampled subgraph from the input graph.
    """

    # Set the random seed for reproducibility
    random.seed(seed)

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    # Convert the sample percent to the number of nodes
    total_nodes = len(original_graph.nodes())
    sample_size = int(total_nodes * sample_percent)

    # Set to track visited nodes to avoid cycling
    visited = set()

    # Get a random start node (ambassador node)
    nodes_list = list(original_graph.nodes())
    start_node = random.choice(nodes_list)

    # BFS-like queue initialized with the start node
    queue = deque([start_node])
    visited.add(start_node)

    # Fire spreads until we reach the sample size or the fire stops
    while len(visited) < sample_size and queue:
        current_node = queue.popleft()

        # Get the neighbors (both in-links and out-links)
        neighbors = list(original_graph.neighbors(current_node))

        # Sample forward-burning neighbors with probability p
        num_forward_burn = min(
            len(neighbors), max(1, np.random.binomial(len(neighbors), p))
        )
        forward_neighbors = random.sample(neighbors, num_forward_burn)

        for neighbor in forward_neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

        # For directed graphs, sample backward neighbors with probability r * p
        if original_graph.is_directed():
            in_neighbors = list(original_graph.predecessors(current_node))
            num_backward_burn = min(
                len(in_neighbors), max(1, np.random.binomial(len(in_neighbors), r * p))
            )
            backward_neighbors = random.sample(in_neighbors, num_backward_burn)

            for neighbor in backward_neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Stop once we've reached the target sample size
        if len(visited) >= sample_size:
            break

    # Create the sample graph from the visited nodes
    sample = original_graph.subgraph(list(visited))

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample_graph)))

    return sample_graph


def get_sample_sff(
    original_graph, sample_percent, p=0.7, r=0.3, max_attempts=5, seed=42
):
    """
    Implements the Spontaneous Forest Fire Sampling (SFFS) method.

    Parameters:
        original_graph (NetworkX graph): The input graph to sample from.
        sample_percent (int): The number of nodes to sample as a percentage.
        p (float): Forward burning probability (0 < p < 1).
        r (float): Backward burning ratio (0 < r < 1).
        max_attempts (int): Maximum spontaneous jumps allowed before giving up.
        seed (int): Random seed for reproducibility.

    Returns:
        NetworkX graph: Sampled subgraph from the input graph.
    """

    # Set the random seed for reproducibility
    random.seed(seed)

    # Directed or Undirected graph
    sample = nx.DiGraph() if original_graph.is_directed() else nx.Graph()

    # Convert the sample percent to the number of nodes
    total_nodes = len(original_graph.nodes())
    sample_size = int(total_nodes * sample_percent)

    # Set to track visited nodes to avoid cycling
    visited = set()

    # Get a random start node (ambassador node)
    nodes_list = list(original_graph.nodes())
    start_node = random.choice(nodes_list)

    # BFS-like queue initialized with the start node
    queue = deque([start_node])
    visited.add(start_node)

    attempts = 0

    while len(visited) < sample_size and attempts < max_attempts:
        if not queue:
            # Spontaneous jump: pick a random new start node that hasn't been visited
            remaining_nodes = list(set(nodes_list) - visited)

            if not remaining_nodes:
                break

            start_node = random.choice(remaining_nodes)
            queue.append(start_node)
            visited.add(start_node)
            attempts += 1
            continue

        current_node = queue.popleft()

        # Get the neighbors (both in-links and out-links) for directed graphs
        if original_graph.is_directed():
            out_neighbors = list(original_graph.successors(current_node))
            in_neighbors = list(original_graph.predecessors(current_node))
        else:
            out_neighbors = list(original_graph.neighbors(current_node))
            in_neighbors = []

        # Combine both in-links and out-links into a single list of neighbors
        combined_neighbors = out_neighbors + in_neighbors

        if combined_neighbors:
            # Sample neighbors with forward-burning probability p
            num_burn = max(1, np.random.binomial(len(combined_neighbors), p))

            # Sample neighbors uniformly from combined_neighbors
            selected_neighbors = random.sample(combined_neighbors, num_burn)

            for neighbor in selected_neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

                # Break the loop if the sample size is reached
                if len(visited) >= sample_size:
                    break

        # Add another break to stop the outer loop if the sample size is reached
        if len(visited) >= sample_size:
            break

    # Create the sample graph from the visited nodes
    sample = original_graph.subgraph(list(visited))

    # Create a copy to avoid frozen graph
    sample_graph = (
        nx.DiGraph(sample) if original_graph.is_directed() else nx.Graph(sample)
    )

    # Remove isolated nodes
    sample_graph.remove_nodes_from(list(nx.isolates(sample_graph)))

    return sample_graph


def get_sample_by_method(
    original_graph, sample_percent, method="rn", seed=42, **kwargs
):
    """
    Centralized function to choose the sampling method based on the given abbreviation.

    :param original_graph: The original graph to sample from.
    :param sample_percent: The percentage of the graph to sample.
    :param method: Abbreviation of the method to use for sampling. (e.g., "rn", "bsf", "sffs")
    :param seed: Random seed for reproducibility.
    :param kwargs: Additional parameters specific to the selected method.
    :return: Sampled graph.
    """

    methods = {
        "rn": get_sample_rn,
        "rdn": get_sample_rdn,
        "rpn": get_sample_rpn,
        "re": get_sample_re,
        "rne": get_sample_rne,
        "hyb": get_sample_hyb,
        "rnn": get_sample_rnn,
        "bsf": get_sample_bsf,
        "dsf": get_sample_dsf,
        "rw": get_sample_rw,
        "rj": get_sample_rj,
        "drn": get_sample_drn,
        "dre": get_sample_dre,
        "drne": get_sample_drne,
        "ff": get_sample_ff,
        "sff": get_sample_sff,
    }

    # Get the correct function based on the method
    if method in methods:
        return methods[method](original_graph, sample_percent, seed=seed, **kwargs)
    else:
        raise ValueError(
            f"Invalid method abbreviation '{method}' provided. Available methods are: {', '.join(methods.keys())}"
        )
