from src.preprocess import DataPreprocessor

processor = DataPreprocessor(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

X_train, X_test, y_train, y_test = processor.preprocess()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)