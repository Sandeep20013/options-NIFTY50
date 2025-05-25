

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer


def train_finbert(train_dataset, val_dataset, output_dir='./finbert-india'):
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# Example usage
# sentences = ["there is a shortage of capital, and we need extra financing",  
#              "growth is strong and we have plenty of liquidity", 
#              "there are doubts about our finances", 
#              "profits are flat"]
# print(predict_sentiment(sentences))