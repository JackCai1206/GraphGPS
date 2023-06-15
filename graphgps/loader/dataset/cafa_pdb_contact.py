import math
import numpy as np
from torch_geometric.data import Dataset, Data, InMemoryDataset
import os
import torch
import torch.nn.functional as F
import h5py
import json

def contact2graph(contact: torch.Tensor, node_feats: torch.Tensor, labels: torch.Tensor, threshold = 10, num_classes=None) -> Data:
    assert contact.shape[0] == node_feats.shape[0]
    graph = Data()
    adj = contact < threshold # (N, N)
    graph.edge_index = adj.nonzero().T # (2, E)
    graph.edge_attr = None # no edge features for now
    graph.x = node_feats
    graph.y = torch.sum(F.one_hot(labels, num_classes), 0,  keepdim=True).float()
    return graph

class CAFA5PDBDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, max_classes=None):
        self.max_classes = max_classes
        super(CAFA5PDBDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self): 
        cont_map_dir = os.path.join(self.root, 'contact-maps')
        t5_data_dir = os.path.join(self.root, ' t5-residual-embeddings')

        self.ds = h5py.File(os.path.join(cont_map_dir, 'train_structures.h5'))
        self.label_data = json.load(open(os.path.join(cont_map_dir, 'label_data.json')))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'contact-maps')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'GraphGPS-processed')

    def process(self):
        data_list = []
        for chain_ids, prot_id, contact_map, labels in zip(self.ds['chain_id'], self.ds['prot_id'], self.ds['contact_maps'], self.ds['labels']):
            # print(chain_ids, prot_id, contact_map.shape, labels.shape)
            chain_ids = chain_ids.decode('utf-8')
            prot_id = prot_id.decode('utf-8')
            n = int(math.sqrt(contact_map.shape[0]))
            contact_map = torch.tensor(contact_map).reshape((n, n))
            labels = torch.tensor(labels)
            node_feats = torch.zeros(contact_map.shape[0], 64)
            labels_fil = labels[labels < self.max_classes].long()
            # if (labels_fil.shape[0] == 0):
            #     continue
            graph = contact2graph(contact_map, node_feats, labels_fil, num_classes=self.max_classes)
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
