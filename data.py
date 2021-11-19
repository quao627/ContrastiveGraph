from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from augs import *

def load_dataset(name):
    root = './'
    dataset = TUDataset(root, name, cleaned=True)
    return dataset

class ContrastiveLearningDataset(Dataset):
    def __init__(self, data, view1="node", view2="edge", transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data = data
        self.length = len(data)
        self.dataset = self.get_dataset(data, view1, view2)

    def get_dataset(self, view1, view2):
        aug_funcs = {"node": drop_nodes,
                     "edge": permute_edges,
                     "subgraph": subgraph,
                     "mask_node": mask_nodes}

        func1 = aug_funcs[view1]
        func2 = aug_funcs[view2]

        self.data_1 = []
        self.data_2 = []

        for graph in self.data:
            graph1 = func1(graph)
            graph2 = func2(graph)
            self.data_1.append(graph1)
            self.data_2.append(graph2)

    def len(self):
        return self.length

    def get(self, index):
        return [self.data_1[index], self.data_2[index]]
