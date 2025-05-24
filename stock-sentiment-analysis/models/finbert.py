

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def predict_sentiment(text):
    return nlp(text)

# Example usage
# sentences = ["there is a shortage of capital, and we need extra financing",  
#              "growth is strong and we have plenty of liquidity", 
#              "there are doubts about our finances", 
#              "profits are flat"]
# print(predict_sentiment(sentences))