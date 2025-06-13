# # frontend/utils/api_helpers.py

# import requests
# import time
# import os

# BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# def fetch_balance_with_retry(customer_id, retries=3, delay=2):
#     """
#     Tries to fetch account balance from the backend with retries.
#     Returns the balance as float.
#     Raises last exception if all retries fail.
#     """
#     for attempt in range(retries):
#         try:
#             response = requests.get(f"{BASE_URL}/account_balance/{customer_id}", ...)
#             response.raise_for_status()
#             return response.json()["balance"]
#         except Exception as e:
#             if attempt < retries - 1:
#                 time.sleep(delay)
#             else:
#                 raise e

import requests
import time
import os

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def fetch_balance_with_retry(customer_id, retries=3, delay=2):
    for attempt in range(retries):
        try:
            url = f"{BASE_URL}/account_balance/{customer_id}"
            print(f"Fetching balance from URL: {url}")  # Debug the URL
            response = requests.get(url)
            response.raise_for_status()
            json_response = response.json()
            print(f"DEBUG Response JSON: {json_response}")  # Debug actual response
            balance = json_response["balance"]
            return float(balance)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

