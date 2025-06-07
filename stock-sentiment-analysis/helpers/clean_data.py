import re
class IndianNewsDataCleaner:
    def __init__(self, df, country='India', label='Sentiment', desc_col='Content'):
        self.df = df.copy()
        self.country = country
        self.label = label
        self.desc_col = desc_col

    def map_sentiment(self):
        self.df[self.label] = self.df[self.label].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
        return self

    def add_country(self):
        self.df['country'] = self.country
        return self

    def clean_text(self):
        def clean(t):
            t = str(t)
            t = re.sub(r'http\S+|www\S+|https\S+', '', t, flags=re.MULTILINE)
            t = re.sub(r'\S+@\S+', '', t)
            t = t.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            t = re.sub(r'\s+', ' ', t)
            t = re.sub(r'[^\w\s,.!?-]', '', t)
            t = t.lower()
            return t.strip()

        self.df[self.desc_col] = self.df[self.desc_col].apply(clean)
        return self



    def filter_data(self):
        self.df = self.df.drop_duplicates(subset=[self.desc_col, self.label])
        self.df = self.df.dropna(subset=[self.desc_col, self.label])
        self.df = self.df[self.df[self.desc_col].str.len() > 20]  # minimum length threshold
        self.df = self.df[[self.desc_col, self.label]]
        return self

    def get_clean_data(self):
        return self.df.rename(columns={self.desc_col: 'text'})