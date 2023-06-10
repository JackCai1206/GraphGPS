from torch_geometric.data import Dataset, Data
import os
import torch

from biotoolbox.contact_map_builder import ContactMapContainer

def contact2graph(contact: torch.Tensor, node_feat) -> Data:
    assert contact.shape[0] == node_feat.shape[0]
    graph = Data()
    graph.edge_index = torch.nonzero(contact).t()
    graph.edge_attr = None # no edge features for now
    graph.x = node_feat
    graph.y = None # TODO: add labels
    return graph

class PDBDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PDBDataset, self).__init__(root, transform, pre_transform)

        data_dir = '../../../data/alphafold-structures'
        self.raw_file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    @property
    def raw_file_names(self):
        return self.raw_file_names

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        for raw_path in self.raw_paths:
            data = Data()

    def len(self):
        pass

    def get(self, idx):
        pass
