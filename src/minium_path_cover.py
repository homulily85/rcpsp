import networkx as nx


def matching_to_path_cover(matching, n):
    # Build DAG-style next pointers
    next_node = [-1] * n
    for u, v in matching.items():
        if u.endswith('_L') and v.endswith('_R'):
            u_idx = int(u[:-2])
            v_idx = int(v[:-2])
            next_node[u_idx] = v_idx

    # Compute indegrees
    indegree = [0] * n
    for v in next_node:
        if v != -1:
            indegree[v] += 1

    # Start nodes have indegree = 0
    start_nodes = [i for i in range(n) if indegree[i] == 0]

    # Reconstruct paths
    paths = []
    for start in start_nodes:
        path = [start]
        while next_node[start] != -1:
            start = next_node[start]
            path.append(start)
        paths.append(path)

    return paths


def minimum_path_cover(g: nx.DiGraph):
    b = nx.DiGraph()
    left = set()
    right = set()

    for u in g.nodes():
        b.add_node(f"{u}_L", bipartite=0)
        b.add_node(f"{u}_R", bipartite=1)
        left.add(f"{u}_L")
        right.add(f"{u}_R")

    for u, v in g.edges():
        b.add_edge(f"{u}_L", f"{v}_R")

    optimal_bipartite_graph = nx.algorithms.bipartite.maximum_matching(b, top_nodes=left)

    return matching_to_path_cover(optimal_bipartite_graph, g.number_of_nodes())
