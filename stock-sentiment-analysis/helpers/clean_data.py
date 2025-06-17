import re
import pandas as pd

class IndianNewsDataCleaner:
    def __init__(self, df, country='India', label='Sentiment', desc_col='news'):
        self.df = df.copy()
        self.country = country
        self.label = label
        self.desc_col = desc_col

    def print_length_stats(self, msg):
        lengths = self.df[self.desc_col].astype(str).apply(len)
        print(f"{msg}:")
        print(f"  Count: {len(lengths)}")
        print(f"  Min length: {lengths.min()}")
        print(f"  Max length: {lengths.max()}")
        print(f"  Mean length: {lengths.mean():.2f}")
        print()

    def split_rows_in_pairs(self):
        self.print_length_stats("Before splitting rows")

        def group_lines(lines):
            # lines is a list of stripped lines
            grouped = []
            for i in range(0, len(lines), 5):
                pair = lines[i:i+2]
                grouped.append("\n".join(pair))  # join with newline or space if you prefer
            return grouped

        # Apply split, strip, then group by 2 lines
        self.df[self.desc_col] = self.df[self.desc_col].astype(str).apply(
            lambda x: group_lines([line.strip() for line in x.split('\n') if line.strip()])
        )

        self.df = self.df.explode(self.desc_col).reset_index(drop=True)
        self.print_length_stats("After splitting rows")
        return self

    def map_sentiment(self):
        self.df[self.label] = self.df[self.label].map({'Fall': 0, 'Neutral': 1, 'Rise': 2})
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

        # Print length stats before filtering by length
        self.print_length_stats("Before filtering by length")

        self.df = self.df[self.df[self.desc_col].str.len() > 60]  # you can adjust threshold here

        # Print length stats after filtering
        self.print_length_stats("After filtering by length")

        self.df = self.df[[self.desc_col, self.label]]
        return self

    def get_clean_data(self):
        return self.df.rename(columns={self.desc_col: 'text'})