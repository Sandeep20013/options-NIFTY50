import pandas as pd
from sqlalchemy import create_engine

def load_financial_news(user, password, host, port, database, table_name='financial_news'):
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_str)
    df = pd.read_sql(table_name, engine)
    return df