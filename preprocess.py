from datasets import load_dataset
import pandas as pd

train = load_dataset("electricsheepafrica/african-diabetes-dataset" , split="train").to_pandas()
validation = load_dataset("electricsheepafrica/african-diabetes-dataset", split="validation").to_pandas()
test = load_dataset("electricsheepafrica/african-diabetes-dataset", split="test").to_pandas()



# Combine all splits into one dataframe
df = pd.concat([train, validation, test], ignore_index=True)
print(f"Total records: {len(df)}")

# Select only our 7 non-clinical features and target variable
features = ['age', 'sex', 'bmi', 'family_history_diabetes', 
            'previous_gdm', 'physically_active', 'has_hypertension']
target = 'diabetes_status'

df = df[features + [target]]
print("\n--- Selected Features ---")
print(df.head())

print("\n--- Missing Values After Selection ---")
print(df.isnull().sum())

print("\n--- Target Distribution ---")
print(df[target].value_counts())

print("\n--- Unique Values Per Feature ---")
for col in features:
    print(f"{col}: {df[col].unique()}")