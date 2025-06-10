import pandas as pd
import joblib
import sqlalchemy
from sklearn.preprocessing import StandardScaler

def evaluate_transactions(engine_url, model_path):
    # Connect to your DB
    engine = sqlalchemy.create_engine(engine_url)

    # Fetch transactions with 'is_fraud' as NULL (pending evaluation)
    query = "SELECT transaction_id, amount, location FROM transactions_qml WHERE is_fraud IS NULL"
    transactions_df = pd.read_sql(query, engine)

    if transactions_df.empty:
        print("No transactions to evaluate.")
        return

    # Prepare features (including encoding location)
    features = pd.get_dummies(transactions_df[['amount', 'location']], drop_first=True)

    # Scaling (assuming your model was trained with scaled data)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Load your trained model
    model = joblib.load(model_path)

    # Predict
    predictions = model.predict(features_scaled)

    # Update DB with predictions
    transactions_df['is_fraud'] = predictions

    # Update database
    with engine.connect() as connection:
        for idx, row in transactions_df.iterrows():
            update_query = """
            UPDATE transactions_qml
            SET is_fraud = :is_fraud
            WHERE transaction_id = :transaction_id
            """
            connection.execute(
                sqlalchemy.text(update_query),
                {'is_fraud': bool(row['is_fraud']), 'transaction_id': row['transaction_id']}
            )

    print("Transactions evaluated and updated successfully.")

if __name__ == "__main__":
    evaluate_transactions(
        "postgresql://user:password@localhost:5432/banking",
        "backend/fraud_detection/models/quantum_qsvm.pkl"
    )
