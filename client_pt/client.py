import flwr as fl
import torch
import numpy as np
from model_pt import Net
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

model = Net()
x_train = torch.rand(100, 10)
y_train = torch.randint(0, 2, (100, 1), dtype=torch.float)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=10)

def get_parameters(): return [val.cpu().numpy() for val in model.state_dict().values()]
def set_parameters(params):
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), params):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config): return get_parameters()
    def fit(self, parameters, config):
        set_parameters(parameters)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        return get_parameters(), len(x_train), {}
    def evaluate(self, parameters, config):
        set_parameters(parameters)
        model.eval()
        loss = F.binary_cross_entropy(model(x_train), y_train)
        return float(loss), len(x_train), {}

fl.client.start_numpy_client(server_address="flwr_server:8080", client=FlowerClient())
