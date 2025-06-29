from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import Base, engine
from app.routers import auth, accounts, customer, employee, transactions

Base.metadata.create_all(bind=engine)
app = FastAPI(title="QBank API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth)
app.include_router(accounts)
app.include_router(customer)
app.include_router(employee)
app.include_router(transactions)