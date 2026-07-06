import os
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
DATA_PATH = "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(DATA_PATH)
print("EDA Script Started...")
print("="*60)
print("DATASET SHAPE")
print("="*60)
print(df.shape)

print("\n")

print("="*60)
print("FIRST FIVE ROWS")
print("="*60)
print(df.head())

print("\n")

print("="*60)
print("COLUMN NAMES")
print("="*60)
print(df.columns.tolist())

print("\n")

print("="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

print("\n")

print("="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

print("\n")

print("="*60)
print("STATISTICAL SUMMARY")
print("="*60)
print(df.describe())

print("\n")

print("="*60)
print("TARGET DISTRIBUTION")
print("="*60)
print(df["Attrition"].value_counts())

print("\n")

print("="*60)
print("TARGET DISTRIBUTION (%)")
print("="*60)
print(df["Attrition"].value_counts(normalize=True)*100)


# -----------------------------
# Create plots directory
# -----------------------------

os.makedirs("../plots", exist_ok=True)

# -----------------------------
# Attrition Count
# -----------------------------

plt.figure(figsize=(6,4))

df["Attrition"].value_counts().plot(kind="bar")

plt.title("Employee Attrition")

plt.xlabel("Attrition")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("../plots/attrition_distribution.png")

plt.close()


# -----------------------------
# Numerical Features
# -----------------------------

numerical_cols = df.select_dtypes(include=["int64","float64"]).columns

for col in numerical_cols:

    plt.figure(figsize=(6,4))

    df[col].hist(bins=20)

    plt.title(col)

    plt.tight_layout()

    plt.savefig(f"../plots/{col}.png")

    plt.close()


# -----------------------------
# Correlation Matrix
# -----------------------------

corr = df[numerical_cols].corr()

plt.figure(figsize=(15,10))

plt.imshow(corr,cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(corr.columns)),corr.columns,rotation=90)

plt.yticks(range(len(corr.columns)),corr.columns)

plt.tight_layout()

plt.savefig("../plots/correlation_matrix.png")

plt.close()


print("\nEDA Completed Successfully!")