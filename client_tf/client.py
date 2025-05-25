import numpy as np
import tensorflow as tf
import flwr as fl
from model_tf import get_model

model = get_model()

# Dummy data
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config): return model.get_weights()
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=10)
        return model.get_weights(), len(x_train), {}
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_train, y_train)
        return loss, len(x_train), {"accuracy": acc}

fl.client.start_numpy_client(server_address="flwr_server:8080", client=FlowerClient())
