DROP VIEW IF EXISTS account_profiles;
DROP TABLE IF EXISTS transaction_keys;
DROP TABLE IF EXISTS transactions_encrypted;
DROP TABLE IF EXISTS transactions_qml;
DROP TABLE IF EXISTS classification_metrics;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS contact_info;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS algorithms;
DROP TABLE IF EXISTS roles;

CREATE TABLE roles
(
  role_id SERIAL PRIMARY KEY,
  role_name TEXT NOT NULL
);

INSERT INTO roles
  (role_name)
VALUES
  ('customer'),
  ('employee'),
  ('admin');

CREATE TABLE accounts
(
  account_id SERIAL PRIMARY KEY,
  encrypted_balance BYTEA NOT NULL,
  username TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  role_id INT NOT NULL
    REFERENCES roles(role_id)
);

CREATE TABLE contact_info
(
  id SERIAL PRIMARY KEY,
  "address" TEXT,
  phone TEXT,
  email TEXT
);

CREATE TABLE customers
(
  customer_id SERIAL PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  encrypted_personal_id BYTEA NOT NULL,
  full_name TEXT NOT NULL,
  account_balance NUMERIC(19,4) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  account_id INT NOT NULL
    REFERENCES accounts(account_id),
  contact_info_id INT
    REFERENCES contact_info(id)
);

-- 5) Employees (tied to an account)
CREATE TABLE employees
(
  employee_id SERIAL PRIMARY KEY,
  account_id INT NOT NULL
    REFERENCES accounts(account_id),
  contact_info_id INT
    REFERENCES contact_info(id),
  full_name TEXT NOT NULL,
  position TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE transactions
(
  transaction_id BIGSERIAL PRIMARY KEY,
  account_id INT NOT NULL
    REFERENCES accounts(account_id)
    ON UPDATE CASCADE,
  from_account_id INT NOT NULL
    REFERENCES accounts(account_id)
    ON UPDATE CASCADE,
  to_account_id INT NOT NULL
    REFERENCES accounts(account_id)
    ON UPDATE CASCADE,
  encrypted_amount BYTEA NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE transactions_encrypted
(
  transaction_id BIGSERIAL PRIMARY KEY,
  customer_id INT NOT NULL
    REFERENCES customers(customer_id)
    ON UPDATE CASCADE,
  amount NUMERIC(15,2) NOT NULL,
  encrypted_data BYTEA NOT NULL,
  timestamp TIMESTAMP NOT NULL DEFAULT now(),
  from_account_id INT
    REFERENCES accounts(account_id)
    ON UPDATE CASCADE,
  to_account_id INT
    REFERENCES accounts(account_id)
    ON UPDATE CASCADE
);

CREATE TABLE transaction_keys
(
  transaction_id INT PRIMARY KEY
    REFERENCES transactions_encrypted(transaction_id),
  bb84_key BYTEA NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE transactions_qml
(
  transaction_id SERIAL PRIMARY KEY,
  amount NUMERIC(19,4) NOT NULL,
  transaction_time TIMESTAMP NOT NULL,
  geo_location TEXT,
  ip_address TEXT,
  is_fraud BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE algorithms
(
  id SERIAL PRIMARY KEY,
  algorithm_name TEXT
);

INSERT INTO algorithms
  (algorithm_name)
VALUES
  ('qsvm'),
  ('vqc');

CREATE TABLE classification_metrics
(
  id SERIAL PRIMARY KEY,
  algorithm_type_id INT
    REFERENCES accounts(account_id)
    ON UPDATE CASCADE,
  accuracy DOUBLE PRECISION NOT NULL,
  "precision" DOUBLE PRECISION NOT NULL,
  recall DOUBLE PRECISION NOT NULL,
  f1_score DOUBLE PRECISION NOT NULL,
  confusion_matrix JSONB NOT NULL,
  roc_curve JSONB NOT NULL
);

DROP VIEW IF EXISTS account_profiles;