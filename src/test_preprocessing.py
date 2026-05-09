
from preprocessing import preprocess_pipeline

# Pipeline bize 5 farklı değer dönüyor
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline("data/dataset.csv")

print("Eğitim Seti Özellikleri (X_train) Boyutu:", X_train.shape)
print("Test Seti Özellikleri (X_test) Boyutu:", X_test.shape)
print("Eğitim Seti Hedefleri (y_train) Boyutu:", y_train.shape)
