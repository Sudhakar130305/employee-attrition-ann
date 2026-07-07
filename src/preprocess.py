import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Drop unnecessary columns
df.drop(
    columns=[
        "EmployeeCount",
        "EmployeeNumber",
        "StandardHours",
        "Over18"
    ],
    inplace=True
)

# Encode target
df["Attrition"] = LabelEncoder().fit_transform(df["Attrition"])

# Encode categorical features
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Save
pd.DataFrame(X_train).to_csv("../data/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("../data/X_test.csv", index=False)

pd.DataFrame(y_train).to_csv("../data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("../data/y_test.csv", index=False)

print("Preprocessing complete.")
    