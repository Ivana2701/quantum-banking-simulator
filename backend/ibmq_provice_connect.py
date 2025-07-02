import os
from dotenv import load_dotenv
from qiskit_ibm_provider import IBMProvider

load_dotenv()  # Loads variables from .env into environment
token = os.getenv("QISKIT_IBM_TOKEN")
IBMProvider.save_account(token, overwrite=True)

#rin once, then->
#then in files just use ->
#from qiskit_ibm_provider import IBMProvider
#provider = IBMProvider()