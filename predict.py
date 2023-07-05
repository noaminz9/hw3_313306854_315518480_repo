import torch.nn.functional as F
import torch
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv
import requests
import os
import pandas as pd


class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_url = self.url.replace(' ', '%20')
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file, status code: {response.status_code}")

        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            f.write(response.content)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])

class GCN2(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, dropout):
        super().__init__()
        torch.manual_seed(1234567)
        self.name = "GCN2"
        self.dropout = dropout
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def get_predictions(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    return pred.tolist()

if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]

    model = GCN2(dataset=dataset, hidden_channels=1024, dropout=0.5)
    model.load_state_dict(torch.load('gcn_model.pth'))

    y_pred = list(get_predictions(model, data))
    indices = [i for i in range(len(y_pred))]
    df = pd.DataFrame({'idx': indices, 'prediction': y_pred})
    df.to_csv('prediction.csv', index=False)
