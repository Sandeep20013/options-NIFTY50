from datasets import load_dataset
import pandas as pd

# Load the dataset
ds = load_dataset("raeidsaqur/NIFTY", split="train")

# Convert to pandas DataFrame
df = ds.to_pandas()

# Save to Excel file
df.to_excel("NIFTY_dataset.xlsx", index=False)

print("Dataset saved to NIFTY_dataset.xlsx")