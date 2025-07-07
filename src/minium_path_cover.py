from typing import List


def minimum_path_cover(n: int, edges: List[List[int]]) -> List[List[int]]:
    """
    Calculate the minimum path cover of a given DAG.
    n: number of nodes in the DAG (nodes labeled 0..n-1)
    edges: list of (from, to) edges
    Returns a list of paths, each path is a list of node IDs.
    Behavior is undefined if the graph has cycles.
    """
    # Build inedges and outedges
    inedges: List[List[int]] = [[] for _ in range(n)]
    outedges: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n) or not (0 <= v < n):
            raise ValueError(f"Invalid edge: ({u}, {v})")
        outedges[u].append(v)
        inedges[v].append(u)

    # Convert DAG to bipartite graph
    bip_graph = to_bipartite_graph(n, inedges, outedges)

    # Compute maximum matching on bipartite graph
    matching = maximal_matching(n, bip_graph)

    # Convert matching to DAG adjacencies for paths
    dag_adj = to_dag(n, matching)

    # Extract paths from DAG
    return calc_paths(n, dag_adj)


def to_bipartite_graph(n: int, inedges: List[List[int]], outedges: List[List[int]]) -> List[
    List[int]]:
    """
    Split each node into left (0..n-1) and right (n..2n-1) vertices
    Left vertices have outgoing edges; right have incoming.
    """
    size = 2 * n
    adj: List[List[int]] = [[] for _ in range(size)]
    for u, vs in enumerate(outedges):
        for v in vs:
            adj[u].append(v + n)
    for v, us in enumerate(inedges):
        for u in us:
            adj[v + n].append(u)
    return adj


def maximal_matching(n: int, edges: List[List[int]]) -> List[int]:
    """
    Find maximum matching in a bipartite graph with 2n vertices (0..2n-1).
    Returns an array `match_to` of length 2n,
    where for left vertex i (0<=i<n), match_to[i] is the matched right vertex or -1.
    For right vertex j (n<=j<2n), match_to[j] is the matched left vertex or -1.
    """
    size = 2 * n
    match_pair: List[int] = [-1] * size  # match_pair[u] = v means u matched to v

    def find_augmenting_path(u: int, visited: List[bool]) -> bool:
        for w in edges[u]:
            if visited[w]:
                continue
            visited[w] = True
            # If w is unmatched or previously matched partner can be rematched
            if match_pair[w] == -1 or find_augmenting_path(match_pair[w], visited):
                match_pair[w] = u
                return True
        return False

    # Greedy augmenting paths from each left vertex
    for u in range(n):
        visited = [False] * size
        find_augmenting_path(u, visited)

    # Build match_to for all vertices
    match_to: List[int] = [-1] * size
    for right, left in enumerate(match_pair):
        if left != -1:
            match_to[left] = right
    return match_to


def to_dag(n: int, match_to: List[int]) -> List[int]:
    """
    Convert bipartite matching to DAG adjacency: for each left node i,
    match_to[i] gives j+n or -1. Return list of length n where
    dag_adj[i] = matched j or -1.
    """
    dag_adj: List[int] = [-1] * n
    for i in range(n):
        j = match_to[i]
        if j != -1:
            dag_adj[i] = j - n
    return dag_adj


def calc_paths(n: int, dag_adj: List[int]) -> List[List[int]]:
    """
    Given a DAG represented by dag_adj where dag_adj[i] is the successor of i or -1,
    extract all maximal paths starting from nodes with indegree 0.
    """
    indegree = [0] * n
    for v in dag_adj:
        if v != -1:
            indegree[v] += 1

    # Start nodes have indegree zero
    starts = [i for i, deg in enumerate(indegree) if deg == 0]
    paths: List[List[int]] = []

    for s in starts:
        path = [s]
        while dag_adj[s] != -1:
            s = dag_adj[s]
            path.append(s)
        paths.append(path)
    return paths
