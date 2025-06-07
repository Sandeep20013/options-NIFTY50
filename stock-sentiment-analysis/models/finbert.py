from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer

def train_finbert(
    train_dataset,
    val_dataset,
    output_dir='./finbert-india',
    model_name='yiyanghkust/finbert-tone',
    num_labels=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_steps=500,
    seed=42
):
    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        logging_steps=logging_steps,
        seed=seed,
        logging_strategy=evaluation_strategy,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # return model and tokenizer if needed for further usage
    return model, tokenizer, metrics