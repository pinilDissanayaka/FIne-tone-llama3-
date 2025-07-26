import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)


try:
    ds = load_dataset("marcelbinz/Psych-101")
    
    for split in ds:
        ds[split].to_csv(f"data/{split}.csv")


except Exception as e:
    print(f"An error occurred: {e}")