from datasets import load_dataset

dataset = load_dataset("domenicrosati/TruthfulQA")
train_df = dataset["train"].to_pandas()
