import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_eval=1.0,
    min_fit_clients=2,
    min_eval_clients=2,
    min_available_clients=2,
    eval_fn=None,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="[::]:8080", 
        config={"num_rounds": 3},
        strategy=strategy,
        enable_dashboard=True,
        )
