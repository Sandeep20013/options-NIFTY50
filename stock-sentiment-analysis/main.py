import numpy as np
import pandas as pd
from helpers.clean_data import IndianNewsDataCleaner
from helpers.tokenizer_indian import tokenize_function
from datasets import Dataset
from sklearn.model_selection import train_test_split
from postgres_scripts.read_data import load_financial_news
import os
user = os.getenv('DB_USER')
password = os.getenv('DB_PASS')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_NAME')
df = load_financial_news(database=database,
                         host=host,
                         password=password,
                         port=port,
                         user=user
                         )
cleaner = IndianNewsDataCleaner(df, country='India', label = 'Sentiment')
df_clean = (
    cleaner
    .map_sentiment()
    .add_country()
    .clean_text()
    .filter_data()
    .get_clean_data()
)
dataset = Dataset.from_pandas(df_clean)
train_df, test_df = train_test_split(df_clean, test_size=0.2, stratify=df_clean['Sentiment'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['Sentiment'], random_state=42)
# Tokenize dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# print(train_tokenized[0])
# Set format for pytorch

