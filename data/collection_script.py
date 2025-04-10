import yfinance as yf
from datetime import datetime, timedelta

# Fetch full historical data for NIFTY 50 index
ticker = "^NSEI"
nse_inception_date = datetime(day=17, month=9,year=2007)
todaysDate = datetime.today() + timedelta(hours=4, minutes=30)

intervals = {
    "1m": todaysDate - timedelta(days = 8),
    "2m": todaysDate - timedelta(days=59),
    "5m": todaysDate - timedelta(days = 59),
    "60m": todaysDate - timedelta(days=729),
    "60m": todaysDate - timedelta(days=729),
    "4h": todaysDate - timedelta(days=729),
    "1d": nse_inception_date,
    "5d": nse_inception_date,
    "1wk": nse_inception_date,
    "5d": nse_inception_date,
    "1wk": nse_inception_date,
    "1mo": nse_inception_date,
    "3mo": nse_inception_date,
}

for interval, date in intervals.items():
    data = yf.download(ticker, start=date, end= todaysDate, interval=interval)
    print(f"\n{interval}")
    print(data.head())
    data.to_csv(f"{ticker}_{interval}.csv")
