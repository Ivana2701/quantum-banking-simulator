# quantum-banking-simulator

conda activate qbank
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000

conda activate qbank
cd frontend
streamlit run app.py


uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
streamlit run frontend/app.py
