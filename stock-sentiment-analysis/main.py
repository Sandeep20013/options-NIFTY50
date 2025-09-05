from mlflow_scripts import mlflow_finbert
import torch

def main():
    print(torch.cuda.is_available())
    params = {
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 5,
        "weight_decay": 0.01, 
        "model_name": "yiyanghkust/finbert-tone"
    }

    mlflow_finbert.run_mlflow_experiment(params, "Finbert_test_3")

if __name__ == "__main__":
    main()