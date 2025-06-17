from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
import pandas as pd

def main():
    load_dotenv(".\.env")

    df = pd.read_excel('NIFTY_dataset.xlsx')
    print("Dataset loaded. Sample data:")
    print(df.head())

    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    database = os.getenv('DB_NAME')

    if not all([user, password, host, port, database]):
        raise Exception("One or more database environment variables are missing!")

    password_escaped = quote_plus(password)
    conn_str = f"postgresql://{user}:{password_escaped}@{host}:{port}/{database}"
    print(f"Connecting to DB at {host}:{port} ...")
    engine = create_engine(conn_str)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database connection successful.")
    except Exception as e:
        print("Failed to connect to the database.")
        raise e

    print("Uploading data to PostgreSQL...")
    df.to_sql('financial_news', engine, if_exists='replace', index=False)
    print("Data uploaded successfully!")

if __name__ == "__main__":
    main()
