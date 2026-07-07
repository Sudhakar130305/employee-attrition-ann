import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

        self.label_encoders = {}
        self.scaler = StandardScaler()

    def encode_categorical_columns(self):

        categorical_columns = self.df.select_dtypes(include=['object']).columns

        for column in categorical_columns:

            if column != "Attrition":

                encoder = LabelEncoder()

                self.df[column] = encoder.fit_transform(self.df[column])

                self.label_encoders[column] = encoder

    def encode_target(self):

        encoder = LabelEncoder()

        self.df["Attrition"] = encoder.fit_transform(self.df["Attrition"])

        self.label_encoders["Attrition"] = encoder

    def split_features_target(self):

        X = self.df.drop("Attrition", axis=1)

        y = self.df["Attrition"]

        return X, y

    def normalize(self, X_train, X_test):

        X_train = self.scaler.fit_transform(X_train)

        X_test = self.scaler.transform(X_test)

        return X_train, X_test

    def preprocess(self):

        self.encode_categorical_columns()

        self.encode_target()

        X, y = self.split_features_target()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        X_train, X_test = self.normalize(
            X_train,
            X_test
        )

        return (
            X_train,
            X_test,
            y_train.values.reshape(-1,1),
            y_test.values.reshape(-1,1)
        )
    
    