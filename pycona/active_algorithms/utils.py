
def is_clique(G, gamma=1):
    """
    Check if the given graph is a clique.

    :param G: A NetworkX graph.
    :param gamma: A multiplier for the number of edges in a clique, default is 1.
    :return: True if the graph is a clique, False otherwise.
    """
    import networkx as nx

    assert isinstance(G, nx.Graph), "is_clique() function needs a graph as an input"
    n = len(G.nodes)
    expected_edges = gamma * n * (n - 1) // 2  # number of edges in a clique with n nodes
    actual_edges = len(G.edges)  # number of edges in the graph
    return actual_edges == expected_edges


def can_be_clique(G, D):
    """
    Check if the given graph can be completed to form a clique by adding edges.

    :param G: A NetworkX graph.
    :param D: A list of edges that can be added to the graph.
    :return: True if the graph can be completed to form a clique, False otherwise.
    """
    import networkx as nx

    assert isinstance(G, nx.Graph), "can_be_clique() function needs a graph as first input"
    n = len(G.nodes)
    expected_edges = n * (n - 1) // 2  # number of edges in a clique with n nodes
    actual_edges = len(G.edges) + len(D)  # number of edges in the graph
    return actual_edges == expected_edges
