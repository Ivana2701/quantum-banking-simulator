# Quantum Banking Simulator

This project provides a quantum-secure banking application utilizing quantum encryption and quantum machine learning for transaction verification and fraud detection. It features an intuitive frontend interface built with Streamlit and a powerful backend powered by FastAPI, PostgreSQL, and Qiskit.

## Project Overview

### Frontend

* **Login System:** Simple authentication with session management.
* **Dashboard:** Displays account metrics such as account balance. And in the newer version a transaction encryption.
The project demonstrates secure transaction encryption using the BB84 Quantum Key Distribution protocol. Transactions between the user and the bank are encrypted with quantum-generated keys, ensuring robust security against quantum computing threats.

* **Transactions:** Verify transactions using quantum algorithms.
* **Model Evaluation:** Execute quantum machine learning models and display evaluation metrics visually.

### Backend

* **Quantum Encryption:** Securely encrypt and decrypt data using post-quantum cryptography (Kyber512 via OQS).
* **Quantum Machine Learning:** Train, evaluate, and deploy quantum-enhanced models (VQC, QSVM) for fraud detection.
* **Data Handling:** PostgreSQL database integration, with efficient data preprocessing and management.

## Requirements

Ensure you have Conda installed, then create the environment using:

```bash
conda env create -f environment.yml
conda activate qbank
```

## Project Setup

### Backend

Run the backend FastAPI server:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

Run the Streamlit frontend:

```bash
cd frontend
streamlit run app.py
```

### Database

Start PostgreSQL and access your banking database:

```bash
psql -U bankuser -d banking
```

## File Structure

```
quantum-banking-simulator/
├── backend/
│   ├── quantum_encryption/
│   ├── quantum_ml/
│   └── transaction_results/
├── frontend/
│   ├── components/
│   │   ├── dashboard.py
│   │   ├── login.py
│   │   ├── model_evaluation.py
│   │   └── transactions.py
│   └── app.py
├── requirements.txt
├── environment.yml
└── README.md
```

## Usage

* **Log in** using your credentials (default: admin/password).
* **Navigate** through the sidebar to access dashboard metrics, transaction verification, and model evaluation.
* **Evaluate Quantum Models:** Select the desired quantum algorithm and either fetch existing metrics or run the model for updated results.

## Dependencies

* Python 3.10
* FastAPI, Streamlit
* PostgreSQL
* Qiskit (quantum computing libraries)
* OQS (Post-quantum encryption)
* scikit-learn, imblearn, pandas, matplotlib, seaborn, SQLAlchemy

## 🔄 Transaction Mode Toggle:

- **Demo Mode**: Educational only. Temporarily stores quantum-generated BB84 keys for visual understanding. **Not secure**.
- **Real Mode**: Production-level security. BB84 keys **never stored or reused**, adhering strictly to quantum encryption security practices.

## Security Notice

Replace default credentials with secure authentication methods for production deployments.

🔑 Credentials:
Customer Login:
Username: customer
Password: customer

Employee Login:
Username: employee
Password: employee

---

For troubleshooting or assistance, ensure all dependencies match those in `environment.yml`. Enjoy exploring quantum-secure banking!

