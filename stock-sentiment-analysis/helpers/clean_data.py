import re
class IndianNewsDataCleaner:
    def __init__(self, df, country = 'India', label = 'Sentiment', X = 'Content'):
        self.df = df
        self.country = country
        self.label = label
        self.X = X

    def map_sentiment(self):
        self.df[self.label] = self.df[self.label].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
        return self

    def add_country(self):
        self.df['country'] = self.country
        return self
    
    def clean_text(self):
        def clean(t):
            t = t.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            t = re.sub(r'\s+', ' ', t)
            return t.strip()
        
        self.df[self.X] = self.df[self.X].astype(str).apply(clean)
        return self
    
    def filter_data(self):
        self.df = self.df[[self.X, self.label]]
        return self

    def get_clean_data(self):
        return self.df
    
