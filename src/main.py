import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.head())
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["Attrition"].value_counts())