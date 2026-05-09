
import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim):
    # Giriş Katmanı
    input_layer = layers.Input(shape=(input_dim,))
    
    # ENCODER (Veriyi sıkıştıran kısım)
    # Nöron sayılarını kademeli olarak azaltıyoruz
    encoded = layers.Dense(16, activation='relu')(input_layer)
    encoded = layers.Dense(8, activation='relu')(encoded)
    
    # BOTTLENECK (En dar boğaz - verinin özeti)
    bottleneck = layers.Dense(4, activation='relu')(encoded)
    
    # DECODER (Veriyi orijinal haline geri açan kısım)
    decoded = layers.Dense(8, activation='relu')(bottleneck)
    decoded = layers.Dense(16, activation='relu')(decoded)
    
    # ÇIKIŞ KATMANI (Giriş boyutuyla aynı olmalı)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # Modeli birleştirme
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    
    # Derlem (Loss fonksiyonu MSE: Hata kareler ortalaması)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder



