# data.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

load_dotenv()

def load_raw():
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    df = pd.read_sql(
        "SELECT amount, EXTRACT(hour FROM transaction_time) AS hour, "
        "location, is_fraud FROM transactions_qml",
        engine
    )
    df['location'] = pd.factorize(df['location'])[0]
    X = df[['amount','hour','location']].to_numpy()
    y = df['is_fraud'].astype(int).to_numpy()
    return X, y

def preprocess(X, y, test_size=0.3, random_state=42, n_pca=2):
    Xs = StandardScaler().fit_transform(X)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        Xs, y, test_size=test_size, stratify=y, random_state=random_state
    )
    Xtr_bal, ytr_bal = SMOTE(random_state=random_state).fit_resample(Xtr_raw, ytr)

    pca = PCA(n_components=n_pca)
    Xtr_pca = pca.fit_transform(Xtr_bal)
    Xte_pca = pca.transform(Xte_raw)

    return Xtr_raw, Xte_raw, ytr, yte, Xtr_bal, ytr_bal, Xtr_pca, Xte_pca

def load_and_preprocess():
    X, y = load_raw()
    _, _, _, yte, _, ytr_bal, Xtr_pca, Xte_pca = preprocess(X, y)
    return Xtr_pca, Xte_pca, ytr_bal, yte
