import networkx as nx
import os
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random


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


def draw_graph(graph):
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
            graph = read_graph_from_edge_list(os.path.join(directory, file))
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
    print("Generating subgraphs of size ", max_size)

    # Generate all subgraphs of the largest size in the list of motifs
    all_subgraphs = generate_subgraphs(graph, max_size)
    print("Generated", len(all_subgraphs), "subgraphs")

    # Initialize a dictionary to store counts for each motif, starting from 1
    motif_counts = {i: 0 for i in range(len(motifs))}

    # Iterate through all subgraphs and motifs to count occurrences
    for subgraph in all_subgraphs:
        for i, motif in enumerate(motifs):
            if len(subgraph.edges()) == len(motif.edges()):
                if nx.is_isomorphic(subgraph, motif):
                    motif_counts[i] += 1

    return motif_counts


def generate_sample_graph(original_graph, sample_size, seed=42):
    """
    Generates a random sample of a graph with sample_size (%).
    Selects a random sample of nodes from the input graph 
    and includes all edges between the sampled nodes.

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
