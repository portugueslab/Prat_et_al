import numpy as np
from scipy.cluster.hierarchy import to_tree


def cluster_id_search(tree):
    nodes_list = []
    if tree.is_leaf():
        nodes_list.append(tree.get_id())
    else:
        nodes_list += cluster_id_search(tree.get_left())
        nodes_list += cluster_id_search(tree.get_right())

    return nodes_list


def find_trunc_dendro_clusters(linkage_mat, dendro):
    tree, branches = to_tree(linkage_mat, rd=True)
    ids = np.empty(linkage_mat.shape[0] + 1, dtype=int)

    for i, clust in enumerate(dendro["leaves"]):
        ids[cluster_id_search(branches[clust])] = i

    return ids