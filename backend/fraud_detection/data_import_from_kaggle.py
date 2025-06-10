import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlalchemy

def import_transactions_from_kaggle(csv_path: str, engine_url: str):
    # Load data from Kaggle CSV
    df = pd.read_csv(csv_path)

    # Generate transaction timestamps
    base_date = datetime.now()
    df['transaction_time'] = df['Time'].apply(lambda x: base_date - timedelta(seconds=int(x)))

    # Assign random locations realistically
    locations = ['Berlin', 'Dubai', 'London', 'Sofia', 'New York', 'Tokyo']
    np.random.seed(42)
    df['location'] = np.random.choice(locations, size=len(df))

    # Map Kaggle class to boolean
    df['is_fraud'] = df['Class'].astype(bool)

    # Keep columns needed by your database schema
    final_df = df.rename(columns={'Amount': 'amount'})[
        ['amount', 'transaction_time', 'location', 'is_fraud']
    ]

    # Connect and insert into the database
    engine = sqlalchemy.create_engine(engine_url)
    final_df.to_sql('transactions_qml', engine, if_exists='append', index=False)

    print("Data import complete!")

if __name__ == "__main__":
    import_transactions_from_kaggle(
        'backend/fraud_detection/transactions/creditcard.csv',
        'postgresql://user:password@localhost:5432/banking'
    )
