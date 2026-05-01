from datasets import load_dataset
import pandas as pd
print ("loading datset")
dataset = load_dataset("electricsheepafrica/african-diabetes-dataset")
df = dataset['train'].to_pandas()
print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Column Names ---")
print(df.columns.tolist())
print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Target Variable Distribution ---")
print(df['diabetes_status'].value_counts())

print("\n--- Dataset Info ---")
print(dataset)

print("\n--- Available Splits ---")
print(dataset.keys())