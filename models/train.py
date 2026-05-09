
import sys
import os
# Üst klasördeki 'src' klasörünü Python'a tanıtıyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_pipeline
from models.autoencoder import build_autoencoder
import tensorflow as tf

def train_model():
    # 1. Veriyi pipeline dan çek
    print("Veri hazırlanıyor...")
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline("data/dataset.csv")

    
    # 2. Sadece normal verileri ayır (Anomali tespiti mantığı)
    X_train_normal = X_train[y_train == 0]
    
    # 3. Model mimarisini kur
    input_dim = X_train_normal.shape[1]
    model = build_autoencoder(input_dim)
    
    print(f"Eğitim başlıyor... Girdi boyutu: {input_dim}")
    
    # 4. Eğitimi çalıştır
    history = model.fit(
        X_train_normal, X_train_normal, # Giriş ve çıkış aynı
        epochs=30, # Şimdilik 30 devir yeterli
        batch_size=32,
        validation_split=0.2, # %20'sini kontrol için ayır
        shuffle=True,
        verbose=1
    )
    
    # 5. Modeli kaydet
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    import joblib
    model.save('saved_models/autoencoder_model.h5')
    print("\nModel başarıyla eğitildi ve 'saved_models/' klasörüne kaydedildi.")
    #scaler i kaydet
    joblib.dump(scaler, 'saved_models/scaler.pkl')

if __name__ == "__main__":
    train_model()
