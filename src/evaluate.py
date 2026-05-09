
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# 'src' klasörünü Python'a tanıtıyoruz (preprocessing dosyasını bulabilmesi için)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess_pipeline

def evaluate_model():
    print("Test verisi yükleniyor...")
    # Veri setinden sadece test verilerini (X_test, y_test) alıyoruz
    _, X_test, _, y_test, _ = preprocess_pipeline("data/dataset.csv")
    
    print("Eğitilmiş model yükleniyor...")
    model = load_model("saved_models/autoencoder_model.h5", compile=False)
    
    print("Test verisi üzerinde tahmin (reconstruction) yapılıyor...")
    predictions = model.predict(X_test)
    
    # Hata Payını (MSE) Hesaplama
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    
    # Eşik Değer (Threshold) Belirleme: Hataların %95'ini normal, kalanını anomali sayıyoruz
    threshold = np.percentile(mse, 95)
    print(f"\nBelirlenen Anomali Eşik Değeri: {threshold:.4f}")

    np.save("saved_models/threshold.npy", threshold)
    
    # Eşikten büyük olanlara 1 (Anomali), küçük olanlara 0 (Normal) diyoruz
    y_pred = [1 if e > threshold else 0 for e in mse]
    
    # Performans Raporunu Yazdır
    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test, y_pred))
    
    # Hata Dağılım Grafiği (Histogram) Çizdir
    plt.figure(figsize=(10,6))
    plt.hist(mse[y_test==0], bins=50, label='Normal İşlemler', alpha=0.5, color='blue')
    plt.hist(mse[y_test==1], bins=50, label='Anomali (Sahtekarlık)', alpha=0.5, color='red')
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label='Eşik (Threshold)')
    plt.title('Modelin Hata Dağılım Grafiği')
    plt.xlabel('Hata Miktarı (MSE)')
    plt.ylabel('İşlem Sayısı')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
