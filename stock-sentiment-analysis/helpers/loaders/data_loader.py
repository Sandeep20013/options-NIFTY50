import os
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from postgres_scripts.read_data import load_financial_news
from helpers.clean_data import IndianNewsDataCleaner
from helpers.tokenizer_indian import tokenize_function
def get_tokenized_datasets(country='India', label='label', test_size=0.2, val_size=0.1, random_state=42):
    # Load env vars

    # user = os.getenv('DB_USER')
    # password = os.getenv('DB_PASS')
    # host = os.getenv('DB_HOST')
    # port = os.getenv('DB_PORT')
    # database = os.getenv('DB_NAME')

    # # Load and clean raw data
    # df = load_financial_news(database=database, host=host, password=password, port=port, user=user)
    df = pd.read_excel("postgres_scripts/NIFTY_dataset.xlsx")
    cleaner = IndianNewsDataCleaner(df, country=country, label=label)
    print(df.columns)

    df_clean = (
        cleaner
        .map_sentiment()
        .split_rows_in_pairs()
        .add_country()
        .clean_text()
        .filter_data()
        .get_clean_data()
    )
    print(df.shape)
    train_df, test_df = train_test_split(df_clean, test_size=test_size, stratify=df_clean[label], random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df[label], random_state=random_state)

    # Tokenize
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)
    
    train_tokenized = train_tokenized.rename_column(label, "labels")
    val_tokenized = val_tokenized.rename_column(label, "labels")
    test_tokenized = test_tokenized.rename_column(label, "labels")

    # Set dataset format to PyTorch tensors
    train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_tokenized, val_tokenized, test_tokenized
