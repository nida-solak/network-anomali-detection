
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def feature_engineering(df):
    if 'Class' in df.columns:
        y = df['Class']
        X = df.drop(columns=['Class'])
    else:
        X = df
        y = None
    return X, y
def split_and_scale(X, y):
    # 1. Önce veriyi Train ve Test olarak bölüyoruz (Data Leakage'ı önlemek için)
    # stratify=y parametresi, anomali oranının train ve test'e eşit dağılmasını sağlar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Uç değerlere dayanıklı RobustScaler kullanıyoruz
    scaler = RobustScaler()
    
    # 3. Scaler'ı SADECE eğitim verisiyle eğitiyoruz (fit)
    X_train_scaled = scaler.fit_transform(X_train)
    # Test verisini sadece dönüştürüyoruz (transform)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def preprocess_pipeline(path):
    df = load_data(path)
    df = clean_data(df)
    X, y = feature_engineering(df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


