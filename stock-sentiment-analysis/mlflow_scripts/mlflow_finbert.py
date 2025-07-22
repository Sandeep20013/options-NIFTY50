import mlflow
import mlflow.transformers
from models.finbert import train_finbert
from helpers.loaders.data_loader import get_tokenized_datasets  # You must implement this
from metrics.eval_metrics import evaluate_finbert
def run_mlflow_experiment(params, exp_name):
    experiment_name = exp_name
    mlflow.set_experiment(experiment_name)
    train_dataset, val_dataset, test_dataset = get_tokenized_datasets()
    
    with mlflow.start_run(run_name="FinBERT_India_1"):
        mlflow.log_params(params)

        model, tokenizer, metrics = train_finbert(
            train_dataset,
            val_dataset,
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params["per_device_eval_batch_size"],
            num_train_epochs=params["num_train_epochs"],
            weight_decay=params["weight_decay"],
            model_name=params["model_name"]
        )
    
        mlflow.log_metrics(metrics)
        test_metrics = evaluate_finbert(test_dataset, model_dir= "./")
        conf_path = test_metrics.pop("confusion_matrix_path")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(conf_path)
        print("Logging model and tokenizer to MLflow...")
        print(f"Model type: {type(model)}, Tokenizer type: {type(tokenizer)}")
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path="finbert-india-model_1",
            tokenizer=tokenizer,
            input_example={"text": "The market outlook is positive"}
        )
