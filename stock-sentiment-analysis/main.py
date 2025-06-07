from mlflow_scripts import mlflow_finbert
import torch

def main():
    print(torch.cuda.is_available())
    params = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "model_name": "yiyanghkust/finbert-tone"
    }

    mlflow_finbert.run_mlflow_experiment(params)

if __name__ == "__main__":
    main()