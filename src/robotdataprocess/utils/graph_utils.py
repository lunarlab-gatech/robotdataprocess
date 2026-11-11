import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from typing import Any, Dict, Hashable, Iterator, Set, Tuple


def _kruskal_merges(edge_weights: Dict[Tuple[Hashable, Hashable], Any]) -> Iterator[Tuple[Tuple[Hashable, Hashable], Set[Hashable], Set[Hashable]]]:
    """
    Kruskal's algorithm: edges are taken in order of (weight, edge), so equal weights are broken
    deterministically by the edges themselves, and an edge is kept only if its two nodes are not
    already connected by edges kept before it.

    Args:
        edge_weights (Dict[Tuple[Hashable, Hashable], Any]): Each edge (node_a, node_b) mapped to its
            weight; weights must be mutually comparable and edges sortable.
    Returns:
        Iterator: For each kept edge, in order, (edge, nodes of node_a's group, nodes of node_b's group)
            as they were just before the edge merged them.
    """

    # Union-find over nodes: each node points towards the root of its connected group, whose nodes are listed by members
    parent: Dict[Hashable, Hashable] = {}
    members: Dict[Hashable, Set[Hashable]] = {}
    def root(node: Hashable) -> Hashable:
        if node not in parent:
            parent[node] = node
            members[node] = {node}
        while parent[node] != node:
            node = parent[node]
        return node

    # Take edges lightest first, keeping only those that connect two separate groups
    for edge in sorted(edge_weights, key=lambda edge: (edge_weights[edge], edge)):
        root_a, root_b = root(edge[0]), root(edge[1])
        if root_a != root_b:
            yield edge, members[root_a], members[root_b]
            parent[root_a] = root_b
            members[root_b] = members[root_b] | members.pop(root_a)


def kruskal_spanning_forest(edge_weights: Dict[Tuple[Hashable, Hashable], Any]) -> Set[Tuple[Hashable, Hashable]]:
    """
    Kruskal's minimum spanning forest of an undirected graph (see _kruskal_merges for the edge order).

    Args:
        edge_weights (Dict[Tuple[Hashable, Hashable], Any]): Each edge (node_a, node_b) mapped to its weight.
    Returns:
        Set[Tuple[Hashable, Hashable]]: The kept edges, which connect the same nodes as the input with no cycles.
    """
    return {edge for edge, _, _ in _kruskal_merges(edge_weights)}


def kruskal_connection_weights(edge_weights: Dict[Tuple[Hashable, Hashable], Any]) -> Dict[Tuple[Hashable, Hashable], Any]:
    """
    For every pair of nodes the graph connects, the weight at which Kruskal's algorithm first connects them:
    the weight of the kept edge that merges their two groups (their minimax path weight).

    Args:
        edge_weights (Dict[Tuple[Hashable, Hashable], Any]): Each edge (node_a, node_b) mapped to its weight;
            nodes must be sortable.
    Returns:
        Dict[Tuple[Hashable, Hashable], Any]: Each connected (node_a, node_b) pair, node_a < node_b, mapped to its
            connection weight. Pairs that are never connected are absent.
    """

    # A kept edge newly connects every node of one merging group with every node of the other
    connection_weights: Dict[Tuple[Hashable, Hashable], Any] = {}
    for edge, group_a, group_b in _kruskal_merges(edge_weights):
        for node_x in group_a:
            for node_y in group_b:
                connection_weights[tuple(sorted((node_x, node_y)))] = edge_weights[edge]
    return connection_weights


def maximum_matching_size(pairs: np.ndarray) -> int:
    """
    Size of a maximum bipartite matching of (left, right) index pairs: the largest subset of the pairs in
    which no left index and no right index repeats.

    Args:
        pairs (np.ndarray): N x 2 array of non-negative integer (left index, right index) pairs; may repeat.
    Returns:
        int: The matching size, which is N when no index repeats on either side.
    """

    # Without a repeated index on either side, every pair can be kept
    num_pairs: int = len(pairs)
    if num_pairs == 0:
        return 0
    if len(np.unique(pairs[:, 0])) == num_pairs and len(np.unique(pairs[:, 1])) == num_pairs:
        return num_pairs

    # Otherwise match on the biadjacency matrix and count the left indices that got a partner
    biadjacency: csr_matrix = csr_matrix((np.ones(num_pairs), (pairs[:, 0], pairs[:, 1])),
                                         shape=(int(pairs[:, 0].max()) + 1, int(pairs[:, 1].max()) + 1))
    partners: np.ndarray = maximum_bipartite_matching(biadjacency, perm_type='column')
    return int(np.sum(partners >= 0))
