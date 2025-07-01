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
or
conda env update -f environment.yml --prune #if you already have an environment
conda activate qbank
``
save exiisting dependencies into requirements.txt`
pipreqs . --force

## Project Setup

### Backend

Always launch the backend first and wait until you see:
INFO:     Application startup complete.

Run the backend FastAPI server:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8002
```
//new project
conda deactivate
conda activate qbank
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000


### Frontend

Run the Streamlit frontend:

```bash
cd frontend
streamlit run app.py
```
//newproject
conda deactivate
conda activate qbank
cd frontend
streamlit run streamlit_app.py

//delete cache
find . -type d -name "__pycache__" -exec rm -r {} +
or
find . -name "*.pyc" -delete


### Database

1. Start PostgreSQL and access your banking database:

```bash
psql -U root -d banking
```
or
```bash
docker run --name banking -p 5432:5432 -e POSTGRES_USER=root -e POSTGRES_PASSWORD=root -d postgres:latest
```
2. Run the quantum-banking-simulator/backend/databse/database_0_increment.sql script in your database

3. Run the quantum-banking-simulator/backend/databse/setup_scripts/create_admin.py sctipt using 
```bash
python quantum-banking-simulator/backend/databse/setup_scripts/create_admin.py
```

### Env setup
Make sure you have .env file in your backend folder:
```bash
# Database Configuration - Set to false to use PostgreSQL (your existing database)
USE_SQLITE=false

# PostgreSQL Configuration (used when USE_SQLITE=false)
DB_NAME=<DB name, usually postgres>
DB_USER=root
DB_PASSWORD=root
DB_HOST=localhost
DB_PORT=5432

# JWT signing key (must match what your FastAPI security.py reads)
JWT_SECRET_KEY=<Secret-key>

# (optional) if you ever want to tweak the algorithm or expiration
#ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=5
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

---

For troubleshooting or assistance, ensure all dependencies match those in `environment.yml`. Enjoy exploring quantum-secure banking!

## Endpoints

1. **New Admin Router** (`backend/app/routers/admin.py`):
   - `/admin/users` - GET endpoint to list all users
   - `/admin/users/{user_id}/role` - PATCH endpoint to update user roles
   - Admin-only access with role verification

2. **New Schemas** (`backend/app/schemas.py`):
   - `RoleUpdateRequest` - For role update requests
   - `RoleUpdateResponse` - For role update responses
