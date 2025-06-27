-- ============================================================================
-- Drop existing tables in dependency order
-- ============================================================================
DROP TABLE IF EXISTS transaction_keys;
DROP TABLE IF EXISTS transactions_encrypted;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS roles;

-- ============================================================================
-- 1) ROLES
-- ============================================================================
CREATE TABLE roles
(
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(20) NOT NULL UNIQUE
);

-- Seed basic roles
INSERT INTO roles
    (role_name)
VALUES
    ('customer'),
    ('employee'),
    ('admin');

-- ============================================================================
-- 2) ACCOUNTS
-- ============================================================================
CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    role_id INT           NOT NULL
        REFERENCES roles(role_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);

-- ============================================================================
-- 3) CUSTOMERS
-- ============================================================================
CREATE TABLE customers
(
    customer_id SERIAL PRIMARY KEY,
    account_id INT NOT NULL UNIQUE
        REFERENCES accounts(account_id)
        ON DELETE CASCADE,
    full_name VARCHAR(100),
    account_balance NUMERIC(15,2) NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 4) EMPLOYEES
-- ============================================================================
CREATE TABLE employees
(
    employee_id SERIAL PRIMARY KEY,
    account_id INT NOT NULL UNIQUE
        REFERENCES accounts(account_id)
        ON DELETE CASCADE,
    full_name VARCHAR(100),
    position VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 5) RAW TRANSACTIONS (encrypted payload)
-- ============================================================================
CREATE TABLE transactions (
    transaction_id VARCHAR(30) PRIMARY KEY,
    account_id INT         NOT NULL
        REFERENCES accounts(account_id)
        ON DELETE RESTRICT,
    encrypted_amount BYTEA       NOT NULL,
    created_at       TIMESTAMP   NOT NULL DEFAULT NOW
()
);

-- ============================================================================
-- 6) BUSINESS‐LEVEL ENCRYPTED TRANSACTIONS
--    (links a customer to a recipient account, stores the ciphertext + clear amount)
-- ============================================================================
CREATE TABLE transactions_encrypted (
    transaction_id VARCHAR(30) PRIMARY KEY
    REFERENCES transactions(transaction_id)
        ON DELETE CASCADE,
    customer_id INT NOT NULL
    REFERENCES customers(customer_id)
        ON DELETE CASCADE,
    recipient_account INT         NOT NULL
        REFERENCES accounts(account_id)
        ON DELETE RESTRICT,
    amount             NUMERIC
(15,2),
    encrypted_data     TEXT        NOT NULL,
    created_at         TIMESTAMP   NOT NULL DEFAULT NOW
()
);

-- ============================================================================
-- 7) KEY‐STORE FOR QUBIT‐BASED TRANSACTION KEYS
-- ============================================================================
CREATE TABLE transaction_keys
(
    transaction_id VARCHAR(30) PRIMARY KEY
        REFERENCES transactions_encrypted(transaction_id)
        ON DELETE CASCADE,
    bb84_key BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- Optional: a handy denormalized VIEW for admin queries
-- ============================================================================
CREATE OR REPLACE VIEW account_profiles AS
SELECT
    a.account_id,
    a.username,
    r.role_name,
    c.full_name       AS customer_name,
    c.account_balance,
    e.full_name       AS employee_name,
    e.position,
    COALESCE(c.created_at, e.created_at, a.created_at) AS created_at
FROM accounts a
    JOIN roles    r ON a.role_id = r.role_id
    LEFT JOIN customers c ON c.account_id = a.account_id
    LEFT JOIN employees e ON e.account_id = a.account_id
WITH NO SCHEMA BINDING;

-- ============================================================================
-- Done!
-- ============================================================================
