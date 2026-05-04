import pandas as pd
import numpy as np

# Column names
columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# Load data from local file
file_path = r"C:\Users\Mohamed Mazhar\Downloads\heart+disease\processed.cleveland.data"
df = pd.read_csv(file_path, names=columns)

# Handling missing values (?)
df.replace("?", np.nan, inplace=True)

# Convert to numeric
df = df.apply(pd.to_numeric)

# Show results
print("Shape:", df.shape)
print(df.head())
df.info()