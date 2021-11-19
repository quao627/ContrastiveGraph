import numpy as np
import torch
from copy import deepcopy


def drop_nodes(data, ratio=0.1):
    data = deepcopy(data)

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * ratio)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data


def permute_edges(data, ratio=0.1):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(ratio * edge_num)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    edge_index = edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)]

    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def subgraph(data, ratio=0.2):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data


def mask_nodes(data, ratio=0.1):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * ratio)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
                                    dtype=torch.float32)

    return data
