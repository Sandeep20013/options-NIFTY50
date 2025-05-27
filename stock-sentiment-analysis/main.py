from mlflow_scripts import mlflow_finbert

def main():
    params = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 4,
        "weight_decay": 0.01,
        "model_name": "yiyanghkust/finbert-tone"
    }

    mlflow_finbert(params)

if __name__ == "__main__":
    main()