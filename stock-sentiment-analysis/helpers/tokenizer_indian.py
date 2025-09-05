from transformers import BertTokenizer
import re
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def tokenize_function(text):
    return tokenizer(text["text"], padding="max_length", truncation=True, max_length=28)
