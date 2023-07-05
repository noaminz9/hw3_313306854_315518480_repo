import requests
import os
from torch_geometric.data import Dataset
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import pickle


os.environ['TORCH'] = torch.__version__
print(torch.__version__)



def visualize(h, color):
    z = TSNE(n_components=3).fit_transform(h.detach().cpu().numpy())
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    plt.xticks([])
    plt.yticks([])
    ax.scatter3D(z[:, 0], z[:, 1], z[:, 2], s=70, c=color, cmap="Set2")
    plt.show()


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


# Models:
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, heads=heads)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


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


class GCN3(torch.nn.Module):
    def __init__(self, hidden_channels, dropout):
        super().__init__()
        torch.manual_seed(1234567)
        self.name = "GCN3"
        self.dropout = dropout
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.conv3 = GCNConv(hidden_channels // 2, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class GCN4(torch.nn.Module):
    def __init__(self, hidden_channels, dropout):
        super().__init__()
        torch.manual_seed(1234567)
        self.name = "GCN4"
        self.dropout = dropout
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        self.conv4 = GCNConv(hidden_channels // 4, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        return x


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask].squeeze(1))  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def validate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[data.val_mask] == data.y[data.val_mask].reshape(-1)  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / len(data.val_mask)  # Derive ratio of correct predictions.
    return val_acc


def train_mlp():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask].squeeze(1))  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test_mlp():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.val_mask] == data.y[data.val_mask].reshape(-1)  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / len(data.val_mask)  # Derive ratio of correct predictions.
    return test_acc


if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]

    # hot encoding year_node onto x
    """
    unique_years = torch.unique(data.node_year)
    num_years = unique_years.size(0)
    num_classes = data.node_year.max() - data.node_year.min() + 1
    dummy_tensor = torch.nn.functional.one_hot(data.node_year - data.node_year.min(), num_classes)
    dummy_tensor = dummy_tensor.view(100000, 48)
    combined_tensor = torch.cat([data.x, dummy_tensor], dim=-1)
    data.x = combined_tensor
    """
    print(data)



    # running MLP model
    """
    model = MLP(hidden_channels=1024)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 301):
        loss = train_mlp()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test_mlp()
    print(f'Test Accuracy: {test_acc:.4f}')"""

    """
    model = GAT(hidden_channels=512, heads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, 401):
        loss = train()
        val_acc = validate()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
    """

    # running GCN model
    model = GCN2(dataset=dataset, hidden_channels=1024, dropout=0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    """train_losses = []
    for epoch in range(1, 301):
        loss = train()
        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    val_acc = validate()
    print(val_acc)"""
    # torch.save(model.state_dict(), 'gcn_model.pth')
    model.load_state_dict(torch.load('gcn_model.pth'))
    out = model(data.x, data.edge_index)
    visualize(out, color=data.y)

    # plot train losses
    """plt.plot([i for i in range(1, 301)], train_losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Train Loss by epoch')
    plt.grid(True)
    plt.savefig('loss_by_epochs.png')
    plt.show()"""

    # Grid Search
    """hidden_list = [512, 1024]
    dropout_list = [0.15, 0.25, 0.5]
    model_list = [GCN2, GCN3, GCN4]
    acc_dict = {}
    for hidden_channels in hidden_list:
        for dropout in dropout_list:
            for model in model_list:
                model = model(hidden_channels=hidden_channels, dropout=dropout)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                criterion = torch.nn.CrossEntropyLoss()
                for epoch in range(1, 301):
                    loss = train()
                    if epoch%10 == 0:
                        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                val_acc = validate()
                acc_dict[(hidden_channels, dropout, model.name)] = val_acc
                print(f'Hidden channels: {hidden_channels}, Dropout: {dropout}'
                      f' Model: {model.name} Validation Accuracy: {val_acc:.4f}')
            with open('acc_dict.pickle', 'wb') as file:
                pickle.dump(acc_dict, file)
    print(acc_dict)
    key_with_largest_value = max(acc_dict, key=acc_dict.get)
    largest_value = acc_dict[key_with_largest_value]

    print("Highest accuracy:", largest_value)
    print("Best paramaters:", key_with_largest_value)
    with open('acc_dict.pickle', 'wb') as file:
        pickle.dump(acc_dict, file) """
    """
    plt.plot(hidden_list, acc_list, marker='o')
    plt.xlabel('Hidden Channels')
    plt.ylabel('Accuracy Scores')
    plt.title('Accuracy Scores by hidden channels size')
    plt.grid(True)
    plt.show()
    plt.savefig('accuracy_by_channels.png')"""


