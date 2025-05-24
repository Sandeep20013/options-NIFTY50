from transformers import BertTokenizer
import re
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def tokenize_function(examples, X='Content'):
    return tokenizer(examples[X], padding="max_length", truncation=True, max_length=128)

