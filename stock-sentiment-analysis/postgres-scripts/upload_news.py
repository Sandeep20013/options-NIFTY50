import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

# Replace with your actual credentials (the ones you used in docker run)
user = os.getenv('DB_USER')
password = os.getenv('DB_PASS')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_NAME')

# Connection string
password_escaped = quote_plus(password)  # This will encode special chars like '@' properly
conn_str = f"postgresql://{user}:{password_escaped}@{host}:{port}/{database}"
# Create SQLAlchemy engine
engine = create_engine(conn_str)

# Load your CSV
df = pd.read_csv("data/Indian_Financial_News.csv")  # adjust path if needed

# Write DataFrame to PostgreSQL table 'financial_news'
df.to_sql('financial_news', engine, if_exists='replace', index=False)

print("CSV data uploaded successfully!")